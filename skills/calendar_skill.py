import urllib.parse

def generate_calendar_link(title: str, start_datetime: str, end_datetime: str, details: str) -> str:
    """
    Call this tool to generate a Google Calendar event link.
    Args:
        title: The title of the event.
        start_datetime: The start time in basic ISO 8601 format (e.g., '20260401T120000Z' or '20260401T120000').
        end_datetime: The end time in basic ISO 8601 format. If none, make it 1 hour after start.
        details: A brief description or location.
    Returns: A URL string.
    """
    title_enc = urllib.parse.quote(title)
    details_enc = urllib.parse.quote(details)
    
    # Strip hyphens/colons if the LLM hallucinated standard ISO format instead of basic
    start = start_datetime.replace("-", "").replace(":", "")
    end = end_datetime.replace("-", "").replace(":", "") if end_datetime else start
    
    url = f"https://calendar.google.com/calendar/render?action=TEMPLATE&text={title_enc}&dates={start}/{end}&details={details_enc}"
    return f"Here is the 1-click link to add this to your calendar: {url}"
