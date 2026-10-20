# Semantic Context Layer (SCL) Language Specification

Version: 1.0

## 1. Overview

SCL (Semantic Context Layer) is a domain-specific XML-like markup language designed to define agent personas, prompt variables, few-shot examples, and structured output expectations in a provider-agnostic manner. SCL documents compile down into native LLM request payloads across supported providers.

---

## 2. Grammar (Informal EBNF)

```ebnf
document   := [ "<scl" scl_attrs ">" ] element* [ "</scl>" ]
element    := identity | variable | example | output

identity   := "<identity" identity_attrs ">" text "</identity>"
variable   := "<variable" var_attrs ">" [ text ] "</variable>"
example    := "<example" example_attrs ">" "<input>" text "</input>" "<output>" text "</output>" "</example>"
output     := "<output" output_attrs ">" text "</output>"

scl_attrs      := [ "version" "=" string ]
identity_attrs := "profile" "=" string [ "tone" "=" string ]
var_attrs      := "name" "=" var_name "source" "=" dotted_path [ "type" "=" ("json" | "text") ]
example_attrs  := [ "pair" "=" ("true" | "false") ]
output_attrs   := "format" "=" ("json_schema" | "markdown") [ "strict" "=" ("true" | "false") ]

var_name       := [a-z_][a-z0-9_]*
dotted_path    := [a-zA-Z_][a-zA-Z0-9_]* ("." [a-zA-Z_][a-zA-Z0-9_]*)*
```

### Element Counts & Constraints
- `<identity>`: **Exactly one** required.
- `<variable>`: **Zero or more**.
- `<example>`: **Zero or more**.
- `<output>`: **At most one**.
- For `<output format="json_schema">`: The tag body contains a raw JSON Schema (Draft-07) and must be treated as raw text until the closing `</output>` tag.

---

## 3. Element Attribute Tables

### `<scl>` (Optional Document Wrapper)
| Attribute | Required | Allowed Values | Default | Description |
|-----------|----------|----------------|---------|-------------|
| `version` | No       | `"1"`, string  | `"1"`   | Language specification version. |

### `<identity>` (Agent Persona Definition)
| Attribute | Required | Allowed Values | Default | Description |
|-----------|----------|----------------|---------|-------------|
| `profile` | **Yes**  | Non-empty string | —     | Functional role or profile name (e.g. `"financial_auditor"`). |
| `tone`    | No       | String         | `None`  | Communication style/tone (e.g. `"formal, analytical"`). |

### `<variable>` (Runtime Data Binding)
| Attribute | Required | Allowed Values | Default | Description |
|-----------|----------|----------------|---------|-------------|
| `name`    | **Yes**  | `[a-z_][a-z0-9_]*` | —     | Variable identifier used in `{{name}}` placeholders. |
| `source`  | **Yes**  | Dotted path    | —       | Dotted resolution path in runtime data (e.g. `runtime.payload.tx`). |
| `type`    | No       | `"json"`, `"text"` | `"text"` | Data serialization/type expectation. |

*Note*: The body of `<variable>` is an optional template that wraps `{{name}}` (e.g. `Transaction Data: {{tx_data}}`).

### `<example>` (Few-Shot Demonstrations)
| Attribute | Required | Allowed Values | Default | Description |
|-----------|----------|----------------|---------|-------------|
| `pair`    | No       | `"true"`, `"false"` | `"true"` | Indicates whether `<input>` and `<output>` form a pair. |

### `<output>` (Output Constraints & Schema)
| Attribute | Required | Allowed Values | Default | Description |
|-----------|----------|----------------|---------|-------------|
| `format`  | **Yes**  | `"json_schema"`, `"markdown"` | — | Target format of agent output. |
| `strict`  | No       | `"true"`, `"false"` | `"false"` | Whether schema enforcement is strict (e.g. `strict: true` for structured outputs). |

---

## 4. Core Rules

1. **Text Preservation**: Inner text content (including whitespace, line breaks, indentation) is preserved verbatim unless explicitly templated.
2. **Placeholder Integrity**: Any `{{placeholder}}` token appearing in text templates must resolve to a declared `<variable name="...">`. Unmatched placeholders result in validation errors.
3. **Identifier Syntax**: Variable names must match regular expression `^[a-z_][a-z0-9_]*$`.
4. **Attribute Validation**: Unknown attributes raise syntax/validation errors unless a lenient parser flag is explicitly passed.
5. **Raw JSON Schema Handling**: If `format="json_schema"`, the parser must not parse inner XML tags; all content up to the matching `</output>` is treated as raw JSON Schema (Draft-07).

---

## 5. Complete Valid Example: Financial Auditor Agent

```xml
<scl version="1">
  <identity profile="financial_auditor" tone="objective, meticulous">
    You are a Senior Forensic Financial Auditor. Your responsibility is to analyze corporate transactions, detect anomalies, verify compliance with GAAP/IFRS standards, and highlight red flags. Maintain professional skepticism and justify every finding with evidence.
  </identity>

  <variable name="transactions" source="runtime.payload.transactions" type="json">
    Target Transactions for Review:
    {{transactions}}
  </variable>

  <variable name="risk_threshold" source="runtime.config.risk_threshold" type="text">
    Materiality / Risk Threshold: {{risk_threshold}}
  </variable>

  <example pair="true">
    <input>
      Transaction ID: TX-1092
      Amount: $450,000.00
      Vendor: ACME Consultants (incorporated 3 days ago)
      Approval: Single-signature override
    </input>
    <output>
      Finding: High-Risk Anomalous Disbursement.
      Risk Score: 0.92
      Details: Vendor creation date immediately precedes large disbursement with single-signature bypass.
    </output>
  </example>

  <output format="json_schema" strict="true">
    {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "type": "object",
      "required": ["findings", "summary", "audit_passed"],
      "properties": {
        "summary": {
          "type": "string"
        },
        "audit_passed": {
          "type": "boolean"
        },
        "findings": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "risk_level", "description"],
            "properties": {
              "id": { "type": "string" },
              "risk_level": { "type": "string", "enum": ["low", "medium", "high", "critical"] },
              "description": { "type": "string" }
            }
          }
        }
      }
    }
  </output>
</scl>
```
