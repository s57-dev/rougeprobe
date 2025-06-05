import os
import json
import inspect
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from functools import wraps
import logging
from jinja2 import Environment, FileSystemLoader, Template
import litellm
from litellm import completion
from enum import Enum
from dotenv import load_dotenv

# Load .env file at import time
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolCallStatus(Enum):
    """Status of tool call execution."""
    SUCCESS = "success"
    ERROR = "error"
    LIMIT_REACHED = "limit_reached"


@dataclass
class ToolCall:
    """Represents a tool call with its metadata."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

@dataclass
class ToolCallResult:
    """Result of a tool call execution."""
    tool_name: str
    result: Any
    status: ToolCallStatus
    error: Optional[str] = None


class ToolCallRegistry:
    """Registry for tool calls."""

    def __init__(self):
        self._tools: Dict[str, ToolCall] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any]):
        """Decorator to register a function as a tool call."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            self._tools[name] = ToolCall(
                name=name,
                description=description,
                parameters=parameters,
                function=func
            )
            return wrapper

        return decorator

    def get_tool(self, name: str) -> Optional[ToolCall]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all_tools(self) -> List[ToolCall]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function format."""
        return [tool.to_openai_function() for tool in self._tools.values()]


tool_registry = ToolCallRegistry()

def toolcall(name: str, description: str, parameters: Dict[str, Any]):
    """Decorator to mark a function as a tool call."""
    return tool_registry.register(name, description, parameters)

class InteractiveAgent:
    """LiteLLM-based interactive agent with tool calling capabilities."""

    def __init__(
            self,
            model: str = "gpt-4",
            prompt_dir: str = "prompts",
            tools_dir: str = "tools",
            max_tool_calls: int = 10,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            **litellm_kwargs
    ):
        """
        Initialize the InteractiveAgent.

        Args:
            model: LiteLLM model name
            prompt_dir: Directory containing J2 prompt templates
            tools_dir: Directory containing tool modules
            max_tool_calls: Maximum number of tool calls per conversation turn
            temperature: Model temperature
            max_tokens: Maximum tokens for response
            **litellm_kwargs: Additional kwargs for LiteLLM
        """
        self.model = model
        self.prompt_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), prompt_dir))
        self.tools_dir = Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), tools_dir))
        self.max_tool_calls = max_tool_calls
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tool_call_counts = {}

        # Load API keys from environment based on model
        self.litellm_kwargs = litellm_kwargs.copy() if litellm_kwargs else {}
        self._set_api_keys_for_model(model)

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompt_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # Load tools
        self._load_tools()

        # Conversation history
        self.messages: List[Dict[str, Any]] = []

        # Tool call counter for current turn
        self.current_tool_calls = 0

    def _set_api_keys_for_model(self, model: str):
        """
        Set the correct API key(s) in self.litellm_kwargs based on the model/provider.
        """
        model_lower = model.lower()
        env = os.environ
        if "openai" in model_lower or model_lower.startswith("gpt-"):
            api_key = env.get("OPENAI_API_KEY")
            if api_key:
                self.litellm_kwargs["api_key"] = api_key
        elif "vertex" in model_lower or "google" in model_lower:
            project = env.get("GOOGLE_PROJECT_ID") or env.get("GOOGLE_PROJECT")
            location = env.get("GOOGLE_LOCATION")
            if project:
                self.litellm_kwargs["google_project_id"] = project
            if location:
                self.litellm_kwargs["google_location"] = location
            # Gemini API key (if needed)
            gemini_key = env.get("GEMINI_API_KEY")
            if gemini_key:
                self.litellm_kwargs["api_key"] = gemini_key
        elif "anthropic" in model_lower or model_lower.startswith("claude"):
            api_key = env.get("ANTHROPIC_API_KEY")
            if api_key:
                self.litellm_kwargs["api_key"] = api_key
        elif "nvidia" in model_lower or "nim" in model_lower:
            api_key = env.get("NVIDIA_NIM_API_KEY")
            if api_key:
                self.litellm_kwargs["api_key"] = api_key
        elif "perplexity" in model_lower or "pplx" in model_lower:
            api_key = env.get("PERPLEXITY_API_KEY")
            if api_key:
                self.litellm_kwargs["api_key"] = api_key
        elif "bedrock" in model_lower or "amazon" in model_lower:
            aws_key = env.get("AWS_ACCESS_KEY_ID")
            aws_secret = env.get("AWS_SECRET_ACCESS_KEY")
            if aws_key and aws_secret:
                self.litellm_kwargs["aws_access_key_id"] = aws_key
                self.litellm_kwargs["aws_secret_access_key"] = aws_secret


    def _load_tools(self):
        """Load all tools from the tools directory."""
        if not self.tools_dir.exists():
            logger.warning(f"Tools directory {self.tools_dir} does not exist")
            return

        # Find all Python files in tools directory
        for py_file in self.tools_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                # Load the module
                spec = importlib.util.spec_from_file_location(
                    py_file.stem, py_file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    logger.info(f"Loaded tools from {py_file.name}")
            except Exception as e:
                logger.error(f"Failed to load tools from {py_file}: {e}")

    def load_system_prompt(self, template_name: str, **kwargs) -> str:
        """
        Load a system prompt from a J2 template.

        Args:
            template_name: Name of the J2 template file
            **kwargs: Parameters to pass to the template

        Returns:
            Rendered prompt string
        """
        try:
            template = self.jinja_env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            raise

    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation."""
        # Remove existing system message if any
        self.messages = [msg for msg in self.messages if msg.get("role") != "system"]
        # Add new system message at the beginning
        self.messages.insert(0, {"role": "system", "content": prompt})

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})

    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolCallResult:
        """Execute a single tool call."""
        tool_name = tool_call.get("function", {}).get("name")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")

        if tool_name:
            self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError as e:
            return ToolCallResult(
                tool_name=tool_name,
                result=None,
                status=ToolCallStatus.ERROR,
                error=f"Failed to parse arguments: {e}"
            )

        tool = tool_registry.get_tool(tool_name)
        if not tool:
            return ToolCallResult(
                tool_name=tool_name,
                result=None,
                status=ToolCallStatus.ERROR,
                error=f"Tool '{tool_name}' not found"
            )

        try:
            # Execute the tool
            result = tool.function(**arguments)
            return ToolCallResult(
                tool_name=tool_name,
                result=result,
                status=ToolCallStatus.SUCCESS
            )
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return ToolCallResult(
                tool_name=tool_name,
                result=None,
                status=ToolCallStatus.ERROR,
                error=str(e)
            )

    def _format_tool_result(self, result: ToolCallResult) -> str:
        """Format tool result for the assistant."""
        if result.status == ToolCallStatus.SUCCESS:
            return json.dumps({
                "tool": result.tool_name,
                "result": result.result
            })
        else:
            return json.dumps({
                "tool": result.tool_name,
                "error": result.error
            })

    async def respond_async(self, user_message: str) -> str:
        """Async version of respond method."""
        return self.respond(user_message)

    def respond(self, user_message: str) -> str:
        """
        Respond to a user message, potentially making tool calls.

        Args:
            user_message: The user's message

        Returns:
            The agent's response
        """
        # Reset tool call counter
        self.current_tool_calls = 0

        # Add user message
        self.add_message("user", user_message)

        # Check if user wants to stop
        if user_message.lower().strip() in ["stop", "quit", "exit"]:
            return "Stopping as requested."

        while True:
            try:
                # Get available tools
                tools = tool_registry.get_openai_functions()

                # Prepare completion kwargs
                completion_kwargs = {
                    "model": self.model,
                    "messages": self.messages,
                    **self.litellm_kwargs
                }

                if self.max_tokens:
                    completion_kwargs["max_tokens"] = self.max_tokens

                if tools and self.current_tool_calls < self.max_tool_calls:
                    completion_kwargs["tools"] = tools


                if "bedrock" not in self.model.lower() and "claude" in self.model.lower():
                    completion_kwargs["tool_choice"] = "auto"

                # Get response from LLM
                response = completion(**completion_kwargs)

                # Extract the message
                message = response.choices[0].message

                # Check if the model wants to make tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Add assistant's message with tool calls
                    self.messages.append(message.model_dump())

                    # Execute tool calls
                    tool_results = []
                    for tool_call in message.tool_calls:
                        if self.current_tool_calls >= self.max_tool_calls:
                            tool_results.append(ToolCallResult(
                                tool_name=tool_call.function.name,
                                result=None,
                                status=ToolCallStatus.LIMIT_REACHED,
                                error="Tool call limit reached"
                            ))
                            break

                        result = self._execute_tool_call(tool_call.model_dump())
                        tool_results.append(result)
                        self.current_tool_calls += 1

                    # Add tool results to conversation
                    for i, result in enumerate(tool_results):
                        self.messages.append({
                            "role": "tool",
                            "content": self._format_tool_result(result),
                            "tool_call_id": message.tool_calls[i].id
                        })

                    # Continue the conversation if not at limit
                    if self.current_tool_calls < self.max_tool_calls:
                        continue

                # No tool calls, return the response
                content = message.content
                self.add_message("assistant", content)
                return content

            except Exception as e:
                logger.error(f"Error in respond: {e}")
                error_msg = f"I encountered an error: {str(e)}"
                self.add_message("assistant", error_msg)
                return error_msg

    def clear_history(self):
        """Clear conversation history except system prompt."""
        self.messages = [msg for msg in self.messages if msg.get("role") == "system"]

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history."""
        return self.messages.copy()

    def save_conversation(self, filepath: str):
        """Save conversation history to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2)

    def load_conversation(self, filepath: str):
        """Load conversation history from a file."""
        with open(filepath, 'r') as f:
            self.messages = json.load(f)