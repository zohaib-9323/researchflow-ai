from __future__ import annotations

import streamlit as st
from ui_common import (
    DOC_STATUS,
    ApiError,
    chip,
    collection_selector,
    esc,
    file_badge,
    friendly_time,
    get_client,
    hero,
    human_size,
    load_sample_documents,
    plural,
    section,
    setup_page,
    stat_cards,
)

client = get_client()
info = setup_page()
state = st.session_state

hero(
    "Knowledge base",
    'Document <span class="rf-grad">library</span>',
    "Add the documents you want to research. Each file is checked, split into passages and "
    "indexed so it can be searched by meaning, not just keywords.",
)

left, right = st.columns([4, 1], vertical_alignment="bottom")
with left:
    collection = collection_selector("library_collection", preselect=state.get("collection_id"))
with (
    right,
    st.popover("New collection", icon=":material/create_new_folder:", use_container_width=True),
    st.form("new_collection", clear_on_submit=True, border=False),
):
    name = st.text_input("Name", max_chars=200, placeholder="e.g. Climate papers")
    if st.form_submit_button("Create", type="primary") and name.strip():
        try:
            created = client.post("/collections", {"name": name.strip()})
        except ApiError as exc:
            st.error(exc.message)
        else:
            state["collection_id"] = created["id"]
            st.rerun()
collection_id = collection["id"] if collection else None
if collection_id:
    state["collection_id"] = collection_id

try:
    documents = client.get("/documents", collection_id=collection_id)
except ApiError as exc:
    st.error(exc.message)
    documents = []
ready = [d for d in documents if d["status"] == "indexed"]
failed = [d for d in documents if d["status"] == "failed"]

# -- overview --------------------------------------------------------------------------------
st.write("")
stat_cards(
    [
        ("Documents", len(documents)),
        ("Ready to search", len(ready)),
        ("Pages", sum(d["page_count"] or 0 for d in ready) or "—"),
        ("Searchable passages", sum(d["chunk_count"] for d in ready)),
    ]
)

# -- upload ----------------------------------------------------------------------------------
section("Add documents", "PDF or TXT · up to 10 files at a time, 20 MB each")
with st.container(key="upload"):
    uploads = st.file_uploader(
        "Drop PDF or TXT files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Scanned PDFs need OCR first.",
    )
    go, sample = st.columns([1, 1])
    start = go.button(
        f"Add {plural(len(uploads), 'file')}" if uploads else "Add files",
        type="primary",
        icon=":material/upload:",
        disabled=not uploads,
        use_container_width=True,
    )
    if sample.button(
        "Load sample documents",
        icon=":material/auto_stories:",
        use_container_width=True,
        help="Three short fictional reports to try the assistant with.",
    ):
        load_sample_documents(collection_id)
        st.rerun()

if start and uploads:
    files = [(f.name, f.getvalue(), f.type or "application/octet-stream") for f in uploads]
    with st.status(f"Processing {plural(len(files), 'file')}…", expanded=True) as progress:
        st.write(
            "Checking files, extracting text, splitting into passages and indexing. "
            "Large PDFs can take a minute."
        )
        try:
            result = client.upload(files, collection_id)
        except ApiError as exc:
            progress.update(label="Upload failed", state="error")
            st.error(exc.message)
        else:
            for item in result["results"]:
                if item["ok"]:
                    doc = item["document"]
                    st.write(
                        f"✅ **{item['filename']}** — ready "
                        f"({plural(doc['chunk_count'], 'passage')})"
                    )
                elif item["error_code"] == "duplicate_document":
                    st.write(f"↩️ **{item['filename']}** — already in this collection, skipped")
                else:
                    st.write(f"❌ **{item['filename']}** — {item['error_message']}")
            progress.update(
                label=f"{result['succeeded']} added, {result['failed']} not added",
                state="complete" if result["failed"] == 0 else "error",
            )
            if result["succeeded"]:
                st.button("Show the updated library", icon=":material/refresh:")

# -- library ---------------------------------------------------------------------------------
head, action = st.columns([3, 1], vertical_alignment="bottom")
with head:
    section("In this collection", plural(len(documents), "document"))
if ready:
    action.page_link(
        "pages/research.py",
        label="Ask about these",
        icon=":material/arrow_forward:",
        use_container_width=True,
    )

if not documents:
    st.info(
        "No documents yet. Add your own files above, or load the sample documents to try "
        "the assistant."
    )
elif len(documents) > 6:
    search = st.text_input(
        "Filter", placeholder="Filter by file name…", label_visibility="collapsed"
    )
    documents = [d for d in documents if search.lower() in d["filename"].lower()]

for doc in documents:
    tone, status_label = DOC_STATUS.get(doc["status"], ("", doc["status"]))
    with st.container(key=f"docrow_{doc['id']}"):
        main, status_col, count_col, action_col = st.columns(
            [6, 1.6, 1.6, 1.4], vertical_alignment="center"
        )
        details = [doc["file_type"].upper(), human_size(doc["size_bytes"])]
        if doc["page_count"]:
            details.append(plural(doc["page_count"], "page"))
        details.append(f"added {friendly_time(doc['created_at'])}")
        main.markdown(
            f'<div class="rf-doc">{file_badge(doc["file_type"])}<div class="rf-doc-main">'
            f'<div class="rf-doc-name" title="{esc(doc["filename"])}">{esc(doc["filename"])}</div>'
            f'<div class="rf-doc-meta">{esc(" · ".join(details))}</div></div></div>',
            unsafe_allow_html=True,
        )
        status_col.markdown(chip(status_label, tone, dot=True), unsafe_allow_html=True)
        if doc["status"] == "indexed":
            count_col.markdown(
                f'<div class="rf-num">{doc["chunk_count"]} '
                f"<small>{'passage' if doc['chunk_count'] == 1 else 'passages'}</small></div>",
                unsafe_allow_html=True,
            )
        confirm_key = f"confirm_delete_{doc['id']}"
        if state.get(confirm_key):
            yes, no = action_col.columns(2)
            if yes.button(
                "",
                key=f"yes_{doc['id']}",
                type="primary",
                icon=":material/delete:",
                help="Permanently remove this document and its passages",
            ):
                try:
                    client.delete(f"/documents/{doc['id']}")
                except ApiError as exc:
                    st.error(exc.message)
                else:
                    state.pop(confirm_key, None)
                    st.toast(f"Removed {doc['filename']}", icon="🗑️")
                    st.rerun()
            if no.button("", key=f"no_{doc['id']}", icon=":material/close:", help="Keep it"):
                state.pop(confirm_key, None)
                st.rerun()
        elif action_col.button(
            "Remove", key=f"del_{doc['id']}", icon=":material/delete:", use_container_width=True
        ):
            state[confirm_key] = True
            st.rerun()
        if state.get(confirm_key):
            st.warning(
                f"Remove **{doc['filename']}**? It will no longer be searchable. "
                "Past answers keep their quotes. Confirm with the red button."
            )
        if doc["status"] == "failed" and doc.get("error_message"):
            st.error(doc["error_message"])

if failed:
    st.caption("Failed files can be fixed and uploaded again — the failed entry is replaced.")
