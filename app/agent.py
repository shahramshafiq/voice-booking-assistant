"""The thinking loop of one answer: ask the model, run the tools it asks for, ask again, until it has spoken.

The loop only produces things to say. It reports them with emit(kind, data):
  ("text", "...")      a piece of the model's answer
  ("tool_start", name) the model has begun a tool request (the arguments are still being written)
  ("tools", [...])     the model asked for tools; the list holds each tool's name and running task
  ("scripted", "...")  text written by the code itself (for example the confirmation question)
The Turn (turn.py) turns these into speech.
"""
import asyncio
import logging

from .tools import for_model

log = logging.getLogger("agent")


def build_instructions(settings):
    """Fill the business details into the prompt template. This text never changes during a conversation,
    so OpenAI can keep it cached, which makes every answer start sooner."""
    services = "\n".join(
        f"- {s['name']} (id {s['id']}): {s['duration_minutes']} minutes. {s.get('description', '')}".rstrip()
        for s in settings.services
    )
    values = {
        "BUSINESS_NAME": settings.business_name,
        "STAFF_NAME": settings.business_staff_name,
        "TIMEZONE": settings.business_timezone,
        "HOURS": settings.business_hours_text.replace(";", "; "),
        "BREAKS": settings.business_breaks_text,
        "MIN_NOTICE": str(settings.min_notice_minutes),
        "MAX_ADVANCE_DAYS": str(settings.max_advance_days),
        "CANCELLATION_NOTICE_HOURS": str(settings.cancellation_notice_hours),
        "SERVICES": services,
    }
    text = settings.system_prompt_template
    for name, value in values.items():
        text = text.replace("{{" + name + "}}", value)
    return text


async def run_agent(*, llm, runner, memory, ctx, settings, instructions, history, user_text, emit, timer):
    tool_items = []  # the tool calls and results of THIS answer, sent back to the model on the next round
    notes = []  # extra facts for the model, for example that a quick "yes" could not be carried out
    running = []  # tasks of tools that have not finished

    def build_input():
        # the stable part (history) first, the changing part (state, question, tool results) last
        state = memory.render_state(runner.service.now(), settings.business_timezone)
        return [*history, {"role": "system", "content": state}, {"role": "user", "content": user_text}, *notes, *tool_items]

    try:
        # A plain "yes" or "no" to the confirmation question needs no model at all.
        decision = runner.quick_decision(ctx)
        if decision == "no":
            await ctx.gate.wait()  # a discarded draft must not forget the question
            memory.pending = None
            await emit("scripted", settings.reply_declined)
            return
        if decision == "yes":
            pending = memory.pending
            await emit("tool_start", pending.tool)  # lets the voice say "I'm on it" at once
            task = asyncio.create_task(runner.run(pending.tool, pending.arguments, ctx))
            running.append(task)
            await emit("tools", [{"name": pending.tool, "task": task}])
            timer.mark("tools_started")
            result = await task
            timer.mark("tools_done")
            if result.get("say"):
                await emit("scripted", result["say"])
                return
            # it could not be done (for example somebody took the time meanwhile): the model explains it
            notes.append({"role": "system", "content": "The caller said yes, but the change could not be made. Result: "
                          + for_model(result) + " Explain this simply and offer a next step."})

        for round_number in range(settings.max_tool_rounds + 1):
            calls = []  # [(call dict, task)]
            last_round = round_number == settings.max_tool_rounds  # on the last round the model must just answer
            async for event in llm.stream(instructions, build_input(), runner.definitions, allow_tools=not last_round):
                if event.kind == "text":
                    await emit("text", event.text)
                elif event.kind == "tool_start":
                    await emit("tool_start", event.call["name"])
                else:
                    # start the tool now, while the model is still finishing its message
                    task = asyncio.create_task(runner.run(event.call["name"], event.call["arguments"], ctx))
                    running.append(task)
                    calls.append((event.call, task))

            if not calls:
                return

            await emit("tools", [{"name": call["name"], "task": task} for call, task in calls])
            timer.mark("tools_started")
            results = await asyncio.gather(*(task for _, task in calls))
            timer.mark("tools_done")

            says = [result["say"] for result in results if result.get("say")]
            if says:  # the code asks the caller to confirm; no need to call the model again
                await emit("scripted", says[0])
                return

            for (call, _), result in zip(calls, results):
                tool_items.append({"type": "function_call", "id": call["id"], "call_id": call["call_id"],
                                   "name": call["name"], "arguments": call["arguments"]})
                tool_items.append({"type": "function_call_output", "call_id": call["call_id"], "output": for_model(result)})

        log.warning("the model kept asking for tools; giving up")
        await emit("scripted", settings.agent_fallback_reply)
    finally:
        for task in running:
            if not task.done():
                task.cancel()