"""Protocol-only RAG double for an isolated Rasa smoke; never a service config."""

from fastapi import FastAPI

app = FastAPI()
requests = []


@app.post("/v1/answer")
async def answer(payload: dict):
    requests.append(payload)
    selection = payload.get("selection")
    items = [{"rank": i, "page_id": f"page:{i}", "target_id": f"chunk:{i}",
              "title": f"Topic {i}", "url": "https://www.tu.berlin/",
              "text": f"Original excerpt {i}"} for i in range(1, 8)]
    if selection:
        items = [item for item in items if item["page_id"] == selection["page_id"]]
    return {"request_id": payload["request_id"], "pipeline_id": "mock:one", "retrieval_adapter": "modern",
            "status": "answered", "answer": "**Test answer:** " + payload["question"], "selection": selection,
            "items": items, "context_target_ids": [i["target_id"] for i in items[:2]]}


@app.get("/requests")
async def recorded():
    return requests
