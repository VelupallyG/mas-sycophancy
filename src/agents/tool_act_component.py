"""ConcatActComponent variant that supports Concordia tool use.

Identical to concordia.components.agent.ConcatActComponent except that
get_action_attempt() creates an InteractiveDocumentWithTools instead of a
plain InteractiveDocument, allowing the LLM to invoke registered tools
during its reasoning loop.

When no tools are provided, this falls back to the exact same behaviour as
ConcatActComponent (InteractiveDocumentWithTools delegates to its base class
when the tool list is empty).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import override

from concordia.document import interactive_document_tools
from concordia.document import tool as tool_module
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class ToolUseActComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
    """Aggregates context from components and supports tool calling."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        tools: Sequence[tool_module.Tool] = (),
        component_order: Sequence[str] | None = None,
        prefix_entity_name: bool = True,
        max_tool_calls_per_question: int = 3,
    ) -> None:
        super().__init__()
        self._model = model
        self._tools = list(tools)
        self._prefix_entity_name = prefix_entity_name
        self._max_tool_calls = max_tool_calls_per_question
        if component_order is None:
            self._component_order = None
        else:
            self._component_order = tuple(component_order)
        if self._component_order is not None:
            if len(set(self._component_order)) != len(self._component_order):
                raise ValueError(
                    "The component order contains duplicate components: "
                    + ", ".join(self._component_order)
                )

    def _context_for_action(
        self, contexts: entity_component.ComponentContextMapping
    ) -> str:
        if self._component_order is None:
            return "\n".join(context for context in contexts.values() if context)
        order = self._component_order + tuple(
            sorted(set(contexts.keys()) - set(self._component_order))
        )
        return "\n".join(contexts[name] for name in order if contexts[name])

    @override
    def get_action_attempt(
        self,
        contexts: entity_component.ComponentContextMapping,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        prompt = interactive_document_tools.InteractiveDocumentWithTools(
            self._model,
            tools=self._tools,
            max_tool_calls_per_question=self._max_tool_calls,
        )
        context = self._context_for_action(contexts)
        prompt.statement(context + "\n")

        call_to_action = action_spec.call_to_action.replace(
            "{name}", self.get_entity().name
        )
        if action_spec.output_type in entity_lib.FREE_ACTION_TYPES:
            output = ""
            if self._prefix_entity_name:
                output = self.get_entity().name + " "
            output += prompt.open_question(
                call_to_action,
                max_tokens=2200,
                answer_prefix=output,
                terminators=(),
                question_label="Exercise",
            )
            self._log(output, prompt)
            return output
        elif action_spec.output_type in entity_lib.CHOICE_ACTION_TYPES:
            idx = prompt.multiple_choice_question(
                question=call_to_action,
                answers=action_spec.options,
            )
            output = action_spec.options[idx]
            self._log(output, prompt)
            return output
        elif action_spec.output_type == entity_lib.OutputType.FLOAT:
            prefix = self.get_entity().name + " " if self._prefix_entity_name else ""
            sampled_text = prompt.open_question(
                call_to_action,
                max_tokens=2200,
                answer_prefix=prefix,
            )
            self._log(sampled_text, prompt)
            try:
                return str(float(sampled_text))
            except ValueError:
                return "nan"
        else:
            raise NotImplementedError(
                f"Unsupported output type: {action_spec.output_type}."
            )

    def _log(
        self,
        result: str,
        prompt: interactive_document_tools.InteractiveDocumentWithTools,
    ) -> None:
        self._logging_channel(
            {
                "Summary": f"Action: {result}",
                "Value": result,
                "Prompt": prompt.view().text().splitlines(),
            }
        )

    def get_state(self) -> entity_component.ComponentState:
        return {
            "component_order": (
                list(self._component_order) if self._component_order else None
            ),
            "prefix_entity_name": self._prefix_entity_name,
        }

    def set_state(self, state: entity_component.ComponentState) -> None:
        if "component_order" in state:
            order = state["component_order"]
            self._component_order = tuple(order) if order else None
        if "prefix_entity_name" in state:
            self._prefix_entity_name = state["prefix_entity_name"]
