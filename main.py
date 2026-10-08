import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request / Response models ----------

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------- Part 1: Execute Python code ----------

def execute_python_code(code: str) -> dict:
    """
    Execute Python code and return exact output.
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr

    stdout = StringIO()
    stderr = StringIO()

    sys.stdout = stdout
    sys.stderr = stderr

    try:
        sandbox_globals = {
            "__name__": "__main__"
        }

        exec(code, sandbox_globals)

        output = stdout.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ---------- Part 2: AI Error Analysis ----------

def analyze_error_with_ai(
    code: str,
    traceback_text: str
) -> List[int]:

    client = OpenAI(
        api_key=os.environ["AIPIPE_TOKEN"],
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
Analyze the Python code and traceback below.

Identify the exact source-code line number or line numbers
where the Python error occurred.

Return ONLY valid JSON in this exact format:

{{
  "error_lines": [3]
}}

Rules:
- Use the source-code line number shown in the traceback.
- Return source-code line numbers, not traceback frame numbers.
- If there is one failing line, return only that line.
- Do not include explanations.
- Do not include markdown.

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            }
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    result = ErrorAnalysis.model_validate_json(
        response.choices[0].message.content
    )

    return result.error_lines


# ---------- FastAPI endpoint ----------

@app.post("/code-interpreter")
def code_interpreter(
    req: CodeRequest
) -> CodeResponse:

    execution = execute_python_code(req.code)

    # Successful execution → DO NOT call AI
    if execution["success"]:
        return CodeResponse(
            error=[],
            result=execution["output"]
        )

    # Error occurred → call AI
    error_lines = analyze_error_with_ai(
        req.code,
        execution["output"]
    )

    return CodeResponse(
        error=error_lines,
        result=execution["output"]
    )

