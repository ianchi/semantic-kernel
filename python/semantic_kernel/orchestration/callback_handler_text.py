from dataclasses import dataclass
from typing import List, Tuple, Union
from semantic_kernel.orchestration.callback_handler_base import CallbackHandlerBase
from semantic_kernel.orchestration.sk_context import SKContext
from semantic_kernel.orchestration.sk_function_base import SKFunctionBase

import html


@dataclass
class FormatTags:
    pipeline_start: Tuple[str, str]
    pipeline_end: Tuple[str, str]

    function_start: Tuple[str, str]
    function_end: Tuple[str, str]
    function_error: Tuple[str, str]
    function_prompt: Tuple[str, str]
    function_prompt_role: Tuple[str, str]
    function_prompt_content: Tuple[str, str]
    line_break: str


class TextHandler(CallbackHandlerBase):
    _clear_on_pipeline: bool
    _content: str
    _sequence: str
    _tags: FormatTags

    def __init__(self, formats: FormatTags, clear_on_pipeline_start=True) -> None:
        self._clear_on_pipeline = clear_on_pipeline_start
        self._content = ""
        self._sequence = ""
        self._tags = formats

    def get_and_reset(self, include_summary=True) -> str:
        content = ""
        if include_summary:
            content = "<h2>Steps Summary</h2>" + self._sequence
            content += "<hr><br><h2>Steps Details</h2>"
        content += self._content
        self._content = ""
        self._sequence = ""

        return content

    @property
    def content(self) -> str:
        return self._content

    def _sanitize(self, text: str) -> str:
        text = html.escape(text)
        text = text.replace("\n", self._tags.line_break)
        return text

    def on_pipeline_start(self, context: SKContext, pipeline_label: str):
        # initialize string
        if self._clear_on_pipeline:
            self._content = ""
            self._sequence = ""

        text = (
            self._tags.pipeline_start[0]
            + self._sanitize(pipeline_label)
            + self._tags.pipeline_start[1]
        )
        self._content += text
        self._sequence += text

    def on_function_start(self, context: SKContext, func: SKFunctionBase):
        if not func.skill_name:
            return
        name = func.skill_name + "." + func.name
        text = self._tags.function_start[0] + self._sanitize(name) + self._tags.function_start[1]
        self._content += text
        self._sequence += text

    def on_prompt_rendered(
        self,
        context: SKContext,
        func: SKFunctionBase,
        prompt: Union[str, List[Tuple[str, str]]],
    ):
        text = ""
        if isinstance(prompt, list):
            for message in prompt:
                text += (
                    self._tags.function_prompt_role[0]
                    + message[0]
                    + self._tags.function_prompt_role[1]
                )
                text += (
                    self._tags.function_prompt_content[0]
                    + message[1]
                    + self._tags.function_prompt_content[1]
                )

        else:
            text = prompt

        self._content += (
            self._tags.function_prompt[0] + self._sanitize(text) + self._tags.function_prompt[1]
        )

    def on_function_end(self, context: SKContext, func: SKFunctionBase):
        if not func.skill_name:
            return
        self._content += (
            self._tags.function_end[0] + self._sanitize(context.result) + self._tags.function_end[1]
        )

    def on_function_error(self, context: SKContext, func: SKFunctionBase):
        name = func.skill_name + "." + func.name
        self._content += self._tags.function_error[0] + self._sanitize(name + "\n")
        self._content += self._sanitize(context.last_error_description)
        if context.last_exception:
            self._content += self._sanitize(str(context.last_exception))
            self._sequence += self._tags.function_error[0] + " Exception in " + self._sanitize(name)
        else:
            self._sequence += (
                self._tags.function_error[0]
                + self._sanitize(name)
                + ": "
                + self._sanitize(context.last_error_description)
            )

        self._content += self._tags.function_error[1]
        self._sequence += self._tags.function_error[1]

    def on_pipeline_end(self, context: SKContext, pipeline_label: str):
        self._content += (
            self._tags.pipeline_end[0] + self._sanitize(pipeline_label) + self._tags.pipeline_end[1]
        )


class HtmlHandler(TextHandler):
    def __init__(self, clear_on_pipeline_start=True) -> None:
        formats = FormatTags(
            pipeline_start=("<h3>Entering Pipeline", "</h3>"),
            pipeline_end=("<br><em>Finished Pipeline", "</em><br><br>"),
            function_start=("<h4>Starting Function ", "</h4>"),
            function_end=("<strong>Function Result:</strong><br>", "<br>"),
            function_error=("<h4 style='color:red'>Error in function</h4> ", "<br>"),
            function_prompt=("<strong>LLM Prompt:</strong><br>", "<br>"),
            function_prompt_role=("<em>", ": </em>"),
            function_prompt_content=("", "<br>"),
            line_break="<br>",
        )
        super().__init__(formats, clear_on_pipeline_start)
