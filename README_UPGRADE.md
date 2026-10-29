# HealthCare AI upgrade

This upgrade preserves the existing Streamlit + Groq + `src.rag` + `src.image_analysis` architecture and adds:

- SQLite persistent conversations and messages
- Per-message timestamps in Asia/Kolkata time
- Saved-chat picker in the sidebar
- Current-chat Markdown and JSON export
- Persistent image description metadata per conversation
- Reduced irrelevant PDF status notices for image-only questions
- Responsive UI polish and clearer project settings

## Apply

1. Back up your current `app.py`.
2. Copy this `app.py` over your project `app.py`.
3. Keep your existing `src/rag.py`, `src/image_analysis.py`, `.env`, vector index, and medical PDFs.
4. Start the existing virtual environment and run `streamlit run app.py`.

No new Python package is required; SQLite ships with Python.

## Storage

The app creates `data/healthcare_chat_history.sqlite3` beside `app.py` on first run. It stores conversation text, timestamps, source metadata, and the generated image description. It does **not** store the uploaded image bytes.

- Local development: chats persist across reruns and browser reloads as long as the database file remains.
- Cloud hosting: ensure the host provides persistent storage, or the database may disappear on redeploy/restart. For multiple app replicas/users, use a managed database and authentication rather than a shared local SQLite file.
- Health privacy: do not store unnecessary identifiers or sensitive health information. Protect the database file and never commit it to Git.

Optional `.gitignore` entries:

```
data/
*.sqlite3
.env
```

Optional timezone override: `APP_TIMEZONE=Asia/Kolkata` (default).

## Important limitation

The image and medical-answer logic still depends on your existing `src/image_analysis.py`, `src/rag.py`, configured Groq models, and API availability. This patch is syntax-checked, but it cannot verify your local credentials, installed dependency versions, vector index, or remote model access from this environment. Run the tests below in your own virtual environment.

## Smoke test

1. Start the app and ask a question; check that both messages show a date/time.
2. Reload the browser; the conversation should still be listed and restored.
3. Start a new conversation; the old one should remain in saved chats.
4. Analyze an image and ask about it; the active description should remain available in that conversation after reload.
5. Ask a medical question and confirm citations use the existing retrieved source metadata.
6. Ask an image-only/non-medical question; it should not show an irrelevant “no medical PDF passages” notice.
7. Export Markdown and JSON; check that timestamps and source metadata are present.
8. Restart Streamlit and confirm the saved chat is still present.
