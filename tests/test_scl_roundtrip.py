"""Hypothesis property-based tests for SCL format and parse round-tripping."""

import json

import hypothesis.strategies as st
from hypothesis import given, settings

from uacp.scl.ast import Document, Example, Identity, Output, Variable
from uacp.scl.parser import format, parse

# Safe text strategy avoiding XML tag delimiters
safe_text_chars = st.characters(
    blacklist_categories=("Cs", "Cc"),
    blacklist_characters="<>&\"'\r\n",
)
st_safe_text = (
    st.text(alphabet=safe_text_chars, min_size=1, max_size=40)
    .map(str.strip)
    .filter(bool)
)

st_var_name = st.from_regex(r"^[a-z_][a-z0-9_]{0,8}$", fullmatch=True)

st_path_segment = st.from_regex(r"^[a-zA-Z_][a-zA-Z0-9_]{0,8}$", fullmatch=True)
st_dotted_path = st.lists(st_path_segment, min_size=1, max_size=3).map(
    lambda parts: ".".join(parts)
)

st_identity = st.builds(
    Identity,
    profile=st_var_name,
    tone=st.one_of(st.none(), st_safe_text),
    body=st_safe_text,
)

st_variable = st.builds(
    Variable,
    name=st_var_name,
    source=st_dotted_path,
    type=st.sampled_from(["text", "json"]),
    template=st.one_of(st.none(), st_safe_text),
)

st_example = st.builds(
    Example,
    pair=st.booleans(),
    input=st_safe_text,
    output=st_safe_text,
)

st_schema = st.fixed_dictionaries(
    {
        "$schema": st.just("http://json-schema.org/draft-07/schema#"),
        "type": st.just("object"),
        "properties": st.fixed_dictionaries(
            {"val": st.fixed_dictionaries({"type": st.just("string")})}
        ),
    }
)


def make_json_output(strict: bool, schema: dict[str, object]) -> Output:
    return Output(
        format="json_schema",
        strict=strict,
        body=json.dumps(schema),
        schema=schema,
    )


st_json_output = st.builds(
    make_json_output,
    strict=st.booleans(),
    schema=st_schema,
)

st_markdown_output = st.builds(
    Output,
    format=st.just("markdown"),
    strict=st.just(False),
    body=st_safe_text,
)

st_output = st.one_of(st_markdown_output, st_json_output)

st_document = st.builds(
    Document,
    version=st.sampled_from(["1"]),
    identity=st_identity,
    variables=st.lists(st_variable, max_size=2),
    examples=st.lists(st_example, max_size=2),
    output=st.one_of(st.none(), st_output),
)


@settings(max_examples=50, deadline=None)
@given(st_document)
def test_hypothesis_roundtrip_document(doc: Document) -> None:
    # Deduplicate variable names to ensure valid SCL
    seen_names: set[str] = set()
    unique_vars: list[Variable] = []
    for v in doc.variables:
        if v.name not in seen_names:
            seen_names.add(v.name)
            unique_vars.append(v)
    doc = Document(
        version=doc.version,
        identity=doc.identity,
        variables=unique_vars,
        examples=doc.examples,
        output=doc.output,
    )

    formatted = format(doc)
    parsed = parse(formatted)
    reformatted = format(parsed)
    reparsed = parse(reformatted)

    assert reparsed == parsed
    assert parse(format(parse(formatted))) == parse(formatted)
