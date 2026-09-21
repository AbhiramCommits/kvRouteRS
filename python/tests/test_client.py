"""SDK client tests (pure parsing; the network path is covered by the
benchmark harness and the router's own integration tests)."""

from kvrouters_sdk import KvRouterClient, iter_chunks


def test_iter_chunks_parses_sse_and_stops_at_done():
    lines = [
        "data: {\"id\": \"a\", \"choices\": [{\"delta\": {\"content\": \"hi\"}}]}",
        "",
        "data: {\"id\": \"a\", \"choices\": [{\"delta\": {\"content\": \" there\"}}]}",
        "data: [DONE]",
        "data: {\"id\": \"b\"}",  # must not be yielded after [DONE]
    ]
    chunks = list(iter_chunks(iter(lines)))
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "hi"
    assert chunks[1]["choices"][0]["delta"]["content"] == " there"


def test_client_builds_request_body():
    client = KvRouterClient("http://127.0.0.1:8080/", model="m")
    body = client._body(
        [{"role": "user", "content": "hello"}],
        stream=True,
        model=None,
        max_tokens=16,
        temperature=None,
        top_p=0.9,
    )
    assert body["model"] == "m"
    assert body["stream"] is True
    assert body["max_tokens"] == 16
    assert "temperature" not in body
    assert body["top_p"] == 0.9
