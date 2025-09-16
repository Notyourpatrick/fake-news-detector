import re

def parse_whatsapp_file(file_path):
    """
    Parse a WhatsApp chat file into structured messages.
    :param file_path: Path to the WhatsApp exported chat text file
    :return: List of messages
    """
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # WhatsApp exported chats usually start with "dd/mm/yyyy, hh:mm - sender: message"
    pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}) - (.*?): (.*)$")

    for line in lines:
        match = pattern.match(line.strip())
        if match:
            date, time, sender, message = match.groups()
            messages.append({
                "date": date,
                "time": time,
                "sender": sender,
                "message": message
            })

    return messages
