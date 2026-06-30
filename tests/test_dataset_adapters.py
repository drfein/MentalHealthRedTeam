from wild_delusion_miner.datasets import iter_user_messages, row_to_conversation


def test_openai_style_conversation_extracts_user_messages():
    row = {
        "conversation_id": "abc",
        "conversation": [
            {"role": "user", "content": "  I can bend time now. "},
            {"role": "assistant", "content": "Tell me more."},
            {"role": "human", "content": "The satellites are answering my thoughts."},
        ],
    }
    conversation = row_to_conversation(row, source="x", split="train", row_offset=3)
    assert conversation is not None
    users = list(iter_user_messages(conversation))
    assert [user.text for user in users] == [
        "I can bend time now.",
        "The satellites are answering my thoughts.",
    ]
    assert users[0].conversation_id == "abc"


def test_sharegpt_style_conversation_extracts_roles():
    row = {
        "id": "def",
        "conversations": [
            {"from": "human", "value": "hello"},
            {"from": "gpt", "value": "hi"},
        ],
    }
    conversation = row_to_conversation(row, source="x", split="train", row_offset=0)
    assert conversation is not None
    assert [(message.role, message.content) for message in conversation.messages] == [
        ("user", "hello"),
        ("assistant", "hi"),
    ]


def test_sharechat_message_row_extracts_top_level_user_message():
    row = {
        "platform": "chatgpt",
        "url": "https://sharechat.example/c/1",
        "turns_count": 4,
        "message_index": 2,
        "role": "user",
        "plain_text": "  The hidden cameras are changing my dreams.  ",
        "topic": "other",
    }
    conversation = row_to_conversation(row, source="sharechat_chatgpt", split="train", row_offset=9)
    assert conversation is not None
    assert conversation.conversation_id == "https://sharechat.example/c/1"
    users = list(iter_user_messages(conversation))
    assert len(users) == 1
    assert users[0].text == "The hidden cameras are changing my dreams."
