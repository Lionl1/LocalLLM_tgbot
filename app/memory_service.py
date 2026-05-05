import logging
from app.llm_client import chat_completion
from app.state import get_knowledge, set_knowledge, persist_knowledge

logger = logging.getLogger(__name__)

KNOWLEDGE_PROMPT = (
    "You are a Memory Manager. Your goal is to maintain a concise Knowledge Base about the users and the conversation.\n"
    "1. Read the CURRENT KNOWLEDGE BASE and the NEW MESSAGES.\n"
    "2. Extract key facts, user preferences, names, habits, and important ongoing context.\n"
    "3. Merge this with the existing Knowledge Base.\n"
    "4. Keep it extremely concise (bullet points). Remove outdated or redundant information.\n"
    "5. Return ONLY the updated Knowledge Base in the same language as the conversation."
)

async def update_knowledge_base(chat_id, new_messages):
    """
    Asynchronously updates the Knowledge Base for a specific chat.
    This should be called in the background to avoid blocking.
    """
    if not new_messages:
        return

    current_kb = get_knowledge(chat_id)
    
    # Format new messages for the LLM
    new_messages_text = ""
    for msg in new_messages:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        new_messages_text += f"{role}: {content}\n"

    prompt = f"CURRENT KNOWLEDGE BASE:\n{current_kb or 'Empty'}\n\nNEW MESSAGES:\n{new_messages_text}"
    
    messages = [
        {"role": "system", "content": KNOWLEDGE_PROMPT},
        {"role": "user", "content": prompt}
    ]

    try:
        logger.info(f"Updating knowledge base for chat {chat_id}...")
        updated_kb = await chat_completion(
            messages,
            max_tokens=256,
            temperature=0.3
        )
        
        if updated_kb and updated_kb.strip():
            set_knowledge(chat_id, updated_kb.strip())
            await persist_knowledge()
            logger.info(f"Knowledge base for chat {chat_id} updated successfully.")
        else:
            logger.warning(f"LLM returned empty knowledge base for chat {chat_id}.")
            
    except Exception as exc:
        logger.error(f"Failed to update knowledge base for chat {chat_id}: {exc}")
