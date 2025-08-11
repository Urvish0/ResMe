import os
import re
from typing import TypedDict, Annotated, List, Literal, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from pylatexenc.latex2text import LatexNodes2Text
from tavily import TavilyClient

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "ResMe-Project"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

class ResumeOptimizationState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    job_description_raw: str
    job_description_text: str
    resume_raw_content: str
    resume_format: Literal["markdown", "latex"]
    resume_plain_text: str
    extracted_keywords: List[str]
    analysis_report: str
    edited_resume_content: str
    human_feedback: str
    next_agent: str
    task_complete: bool
    current_task: str
    old_ats_score: Optional[int]
    new_ats_score: Optional[int]
    cover_letter_text: str
    cover_letter_markdown: str
    cover_letter_analysis: str

@tool
def get_url_content_from_tavily(url: str) -> str:
    """
    Uses Tavily Search to get the content of a specific URL.
    This is specifically designed for scraping web page content.
    """
    try:
        response = tavily_client.get_content(urls=[url])
        if response and response[0]:
            return response[0]
        return f"No content found for URL: {url}"
    except Exception as e:
        return f"Error using Tavily to get URL content: {e}"

@tool
def parse_markdown_to_plain_text(md_content: str) -> str:
    """
    Converts markdown content into plain text, stripping formatting.
    """
    lines = [line.strip() for line in md_content.split('\n') if line.strip() and not line.startswith('#')]
    return "\n".join(lines)

@tool
def extract_text_from_latex(latex_content: str) -> str:
    """
    Extracts plain text from LaTeX content.
    """
    try:
        l2t = LatexNodes2Text()
        plain_text = l2t.latex_to_text(latex_content)
        return plain_text
    except Exception as e:
        return f"Error extracting text from LaTeX: {e}"

def ingestion_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description_raw = state["job_description_raw"]
    resume_raw_content = state["resume_raw_content"]
    resume_format = state["resume_format"]

    messages.append(HumanMessage(content="Starting ingestion process."))
    messages.append(AIMessage(content="Node: `ingestion_node` - Processing raw inputs."))

    job_description_text = ""
    if job_description_raw.startswith("http"):
        messages.append(AIMessage(content=f"Sub-task: Scraping job description from URL: {job_description_raw} using Tavily."))
        scraped_content = get_url_content_from_tavily.invoke({"url": job_description_raw})
        if "Error" in scraped_content or "No content found" in scraped_content:
            messages.append(AIMessage(content=f"Warning: Failed to scrape URL with Tavily. Using raw input as fallback. Error: {scraped_content}"))
            job_description_text = job_description_raw
        else:
            job_description_text = scraped_content
            messages.append(AIMessage(content="Sub-task: Successfully scraped job description content."))
    else:
        job_description_text = job_description_raw
        messages.append(AIMessage(content="Sub-task: Using provided job description text directly."))

    resume_plain_text = ""
    if resume_format == "markdown":
        messages.append(AIMessage(content="Sub-task: Parsing resume from Markdown to plain text."))
        resume_plain_text = parse_markdown_to_plain_text.invoke({"md_content": resume_raw_content})
    elif resume_format == "latex":
        messages.append(AIMessage(content="Sub-task: Extracting plain text from LaTeX resume."))
        resume_plain_text = extract_text_from_latex.invoke({"latex_content": resume_raw_content})
    else:
        messages.append(AIMessage(content="Error: Unsupported resume format provided. Please provide 'markdown' or 'latex'. Ending workflow."))
        return {**state, "messages": messages, "next_agent": END, "task_complete": True}

    messages.append(SystemMessage(content="Node: `ingestion_node` - Job description and resume ingested and converted to plain text."))
    return {
        **state,
        "job_description_text": job_description_text,
        "resume_plain_text": resume_plain_text,
        "messages": messages,
        "next_agent": "keyword_extraction",
        "current_task": "Extracting keywords"
    }

def keyword_extraction_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]

    messages.append(HumanMessage(content="Node: `keyword_extraction_node` - Initiating keyword extraction from job description."))

    prompt = (
        "You are an expert keyword extractor. "
        "Analyze the following job description and identify the most important skills, technologies, and responsibilities. "
        "List them as comma-separated values. Focus on actionable keywords that would be used in a resume.\n\n"
        f"Job Description:\n{job_description}\n\n"
        "Keywords (comma-separated):"
    )
    messages.append(AIMessage(content=f"Sub-task: Sending prompt to LLM for keyword extraction. Prompt snippet: '{prompt[:100]}...'"))
    response = llm.invoke(prompt)
    keywords = [kw.strip() for kw in response.content.split(',') if kw.strip()]

    messages.append(AIMessage(content=f"Sub-task: LLM extracted keywords: {', '.join(keywords)}"))
    messages.append(SystemMessage(content="Node: `keyword_extraction_node` - Keywords extracted successfully."))
    return {
        **state,
        "extracted_keywords": keywords,
        "messages": messages,
        "next_agent": "resume_analysis",
        "current_task": "Analyzing resume"
    }

def resume_analysis_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    resume_text = state["resume_plain_text"]
    keywords = state["extracted_keywords"]
    old_ats_score = None

    messages.append(HumanMessage(content="Node: `resume_analysis_node` - Starting resume analysis against job description and keywords."))

    prompt = (
        "You are a professional resume analyst. "
        "Compare the following resume content with the job description and the extracted keywords. "
        "Provide a detailed report on:\n"
        "**ATS Score: [Your Estimated Score 0-100%]**\n"
        "1. Missing keywords/skills from the resume that are present in the JD.\n"
        "2. Areas where the resume can be strengthened to better align with the JD.\n"
        "3. Suggestions for rephrasing or adding content to highlight relevant experience.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Extracted Keywords:\n{', '.join(keywords)}\n\n"
        f"Resume Content:\n{resume_text}\n\n"
        "Analysis Report:"
    )
    messages.append(AIMessage(content=f"Sub-task: Sending prompt to LLM for initial resume analysis. Prompt snippet: '{prompt[:100]}...'"))
    response = llm.invoke(prompt)
    analysis_report = response.content

    score_match = re.search(r"ATS Score:\s*(\d+)%", analysis_report)
    if score_match:
        old_ats_score = int(score_match.group(1))
        messages.append(AIMessage(content=f"Sub-task: Estimated Original ATS Score: {old_ats_score}%"))
    else:
        messages.append(AIMessage(content="Sub-task: Could not parse original ATS Score from LLM response."))

    messages.append(AIMessage(content=f"Sub-task: Initial resume analysis report generated: \n{analysis_report[:500]}...")) # Truncate for log
    messages.append(SystemMessage(content="Node: `resume_analysis_node` - Resume analysis completed. Moving to human review."))
    return {
        **state,
        "analysis_report": analysis_report,
        "messages": messages,
        "old_ats_score": old_ats_score,
        "next_agent": "human_review", # This is just a label for the current agent's intention
        "current_task": "Awaiting human review (automated)"
    }

def resume_editing_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    resume_text = state["resume_plain_text"]
    analysis_report = state["analysis_report"]
    job_description = state["job_description_text"]
    human_feedback = state["human_feedback"]

    messages.append(HumanMessage(content="Node: `resume_editing_node` - Generating professionally enhanced version of the resume."))

    # Simplified, more direct prompt
    editing_instructions = f"""Improve this resume to make it more professional and ATS-friendly while keeping all original information accurate.

RULES:
- Keep all dates, companies, and factual information exactly the same
- Use stronger action verbs and more professional language
- Better align with job description keywords
- Output ONLY the improved resume in markdown format
- Do not add explanatory text before or after the resume

Original Resume:
{resume_text}

Job Description Keywords to Consider:
{', '.join(state.get('extracted_keywords', []))}

Improved Resume:"""

    if human_feedback and human_feedback.lower() != 'proceed':
        messages.append(AIMessage(content=f"Sub-task: Incorporating human feedback: '{human_feedback}'"))
        editing_instructions += f"\n\nAdditional Instructions: {human_feedback}"

    messages.append(AIMessage(content="Sub-task: Sending enhanced prompt to LLM for professional rewriting."))
    
    try:
        response = llm.invoke(editing_instructions)
        raw_response = response.content.strip()
        
        # Debug output
        print(f"\nDEBUG - Raw LLM Response Length: {len(raw_response)}")
        print(f"DEBUG - First 300 chars of response:")
        print(f"'{raw_response[:300]}...'")
        
        # Simple cleaning - just remove common intro phrases
        edited_resume = clean_resume_response(raw_response)
        
        print(f"DEBUG - Cleaned Response Length: {len(edited_resume)}")
        print(f"DEBUG - First 200 chars of cleaned response:")
        print(f"'{edited_resume[:200]}...'")
        
        # Safety check
        if len(edited_resume) < 50:
            print("WARNING: Cleaned response is too short, using original response")
            edited_resume = raw_response
            
        # Final safety check - if still empty, use original resume with basic improvements
        if len(edited_resume) < 50:
            print("ERROR: LLM returned empty response, using original resume")
            edited_resume = resume_text  # Fallback to original
            
    except Exception as e:
        print(f"ERROR in resume editing: {e}")
        edited_resume = resume_text  # Fallback to original
        messages.append(AIMessage(content=f"Error in LLM call: {e}. Using original resume."))
    
    # Post-processing to ensure no hallucinations were added
    # Commenting out for now as it might be too aggressive
    # edited_resume = remove_added_content(edited_resume, resume_text)
    
    messages.append(AIMessage(content="Sub-task: Professionally enhanced resume content generated."))
    messages.append(SystemMessage(content="Node: `resume_editing_node` - Resume professionally enhanced. Moving to final ATS analysis."))
    
    return {
        **state,
        "edited_resume_content": edited_resume,
        "messages": messages,
        "next_agent": "final_ats_analysis",
        "current_task": "Analyzing new ATS score"
    }

def clean_resume_response(response: str) -> str:
    """
    Simple cleaning function that removes common intro phrases but is conservative.
    """
    response = response.strip()
    
    # Remove common intro lines (only if they're at the very beginning)
    intro_phrases = [
        "Here's the improved professional resume in markdown format:",
        "Here is the improved professional resume:",
        "Improved Professional Resume:",
        "The improved resume:",
        "Here's the improved resume:",
        "Improved Resume:",
    ]
    
    for phrase in intro_phrases:
        if response.lower().startswith(phrase.lower()):
            response = response[len(phrase):].strip()
            break
    
    # Remove markdown code blocks if they wrap everything
    if response.startswith("```markdown"):
        response = response[11:].strip()
    elif response.startswith("```"):
        response = response[3:].strip()
        
    if response.endswith("```"):
        response = response[:-3].strip()
    
    return response
def remove_added_content(edited: str, original: str) -> str:
    """
    Basic safety check to remove any obvious new sections that weren't in original.
    This is a simple implementation - you might want to enhance it further.
    """
    original_sections = set(re.findall(r'^#+\s+.+', original, flags=re.MULTILINE))
    edited_lines = edited.split('\n')
    cleaned_lines = []
    
    current_section = None
    for line in edited_lines:
        # Check if this is a section header
        if re.match(r'^#+\s+.+', line):
            if line not in original_sections:
                current_section = "REMOVE"
            else:
                current_section = None
        
        if current_section != "REMOVE":
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def final_ats_analysis_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    edited_resume_text = state["edited_resume_content"]
    keywords = state["extracted_keywords"]
    new_ats_score = None

    messages.append(HumanMessage(content="Node: `final_ats_analysis_node` - Performing final ATS analysis on the optimized resume."))

    prompt = (
        "You are a professional resume analyst. "
        "Evaluate the following **optimized resume** against the job description and extracted keywords. "
        "Your primary task is to provide an estimated ATS score for this optimized resume.\n"
        "**ATS Score: [Your Estimated Score 0-100%]**\n"
        "Briefly summarize the improvements made in this version relative to the job description requirements."
        "Do NOT provide a full analysis report, only the score and a brief summary of improvements.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Extracted Keywords:\n{', '.join(keywords)}\n\n"
        f"Optimized Resume Content:\n{edited_resume_text}\n\n"
        "Analysis of Optimized Resume:"
    )
    messages.append(AIMessage(content=f"Sub-task: Sending prompt to LLM for final ATS score. Prompt snippet: '{prompt[:100]}...'"))
    response = llm.invoke(prompt)
    new_analysis_summary = response.content

    score_match = re.search(r"ATS Score:\s*(\d+)%", new_analysis_summary)
    if score_match:
        new_ats_score = int(score_match.group(1))
        messages.append(AIMessage(content=f"Sub-task: Estimated Optimized ATS Score: {new_ats_score}%"))
    else:
        messages.append(AIMessage(content="Sub-task: Could not parse new ATS Score from LLM response."))

    messages.append(AIMessage(content=f"Sub-task: Final analysis summary: \n{new_analysis_summary[:500]}..."))
    messages.append(SystemMessage(content="Node: `final_ats_analysis_node` - Final ATS analysis completed."))
    return {
        **state,
        "new_ats_score": new_ats_score,
        "messages": messages,
        "next_agent": "final_response",
        "current_task": "Finalizing output"
    }

def human_review_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    """
    Simulates human review and automatically 'approves' to proceed.
    The actual interrupt() is commented out for automated execution.
    """
    messages = state["messages"]
    analysis_report = state["analysis_report"]

    # Simulating the human review by printing the report and auto-setting feedback
    messages.append(AIMessage(content=f"Node: `human_review_node` - Analysis report for human review:\n{analysis_report}\n\n"))
    messages.append(AIMessage(content="Simulating human review: Automatically setting feedback to 'proceed'."))
    
    # The actual interrupt() for human interaction would go here if not automating:
    # human_prompt_data = {"analysis_report": analysis_report, "message": "Analysis is complete. Please review and provide feedback, or type 'proceed' to continue."}
    # human_response_from_ui = interrupt(human_prompt_data)
    # feedback_text = human_response_from_ui if isinstance(human_response_from_ui, str) else ""

    # For this automated version, we simply set feedback to "proceed"
    feedback_text = "proceed"

    messages.append(SystemMessage(content="Node: `human_review_node` - Human review (automated) completed. Proceeding."))

    return {
        **state,
        "human_feedback": feedback_text,
        "messages": messages,
        # IMPORTANT: Remove "next_agent" from here. This node just updates state.
        # The routing decision is made by the `determine_next_step` router function.
        "current_task": "Processing human decision (automated)"
    }

def determine_next_step(state: ResumeOptimizationState) -> Literal["resume_editing", END]:
    feedback = state.get("human_feedback", "").lower().strip()
    if feedback == "exit" or feedback == "done":
        return END
    else: # If "proceed" or any other feedback (due to automation), it goes to editing
        return "resume_editing"

def save_resume_to_markdown(resume_content: str, filename_prefix: str = "optimized_resume") -> str:
    """
    Saves the optimized resume content to a markdown file with a timestamp.
    Returns the path to the saved file.
    """
    import datetime
    import os
    
    # Create an 'outputs' directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.md"
    filepath = os.path.join("outputs", filename)
    
    # Write the content to file
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(resume_content)
    
    return filepath

def cover_letter_analysis_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    resume_text = state["resume_plain_text"]

    messages.append(HumanMessage(content="Node: `cover_letter_analysis_node` - Analyzing resume for cover letter content extraction."))

    prompt = (
        "Analyze this resume and job description to identify key elements for a cover letter:\n"
        "1. Most relevant 2-3 professional experiences\n"
        "2. Education highlights (if relevant)\n"
        "3. Notable achievements/skills that match the JD\n"
        "4. Professional tone indicators from the resume\n\n"
        "Output ONLY a bullet-point list of these elements.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Resume Content:\n{resume_text}"
    )
    
    response = llm.invoke(prompt)
    analysis = response.content
    return {
        **state,
        "cover_letter_analysis": analysis,
        "messages": messages,
        "next_agent": "cover_letter_generation",
        "current_task": "Generating cover letter"
    }

def cover_letter_generation_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    job_description = state["job_description_text"]
    resume_analysis = state["cover_letter_analysis"]
    edited_resume = state.get("edited_resume_content", "")

    messages.append(HumanMessage(content="Node: `cover_letter_generation_node` - Generating professional cover letter."))

    prompt = (
        "Create a SINGLE-PARAGRAPH professional cover letter using these rules:\n"
        "1. Length: 4-6 concise sentences that fit on one page\n"
        "2. Structure:\n"
        "   - Opening: Who you are and position you're applying for\n"
        "   - Value Proposition: Why you're a strong fit (2-3 key points)\n"
        "   - Closing: Desire to discuss further and gratitude\n"
        "3. Content Rules:\n"
        "   - Use ONLY information from the resume analysis\n"
        "   - Mirror the resume's professional tone\n"
        "   - Include implied contact details (no need to state them)\n"
        "4. Format: Output in markdown with bold section headers\n\n"
        "Resume Analysis:\n"
        f"{resume_analysis}\n\n"
        "Job Description:\n"
        f"{job_description}\n\n"
        "Generated Cover Letter (markdown format):\n"
    )
    
    response = llm.invoke(prompt)
    cover_letter_md = response.content
    
    # Save to file
    cover_letter_path = save_cover_letter_to_markdown(cover_letter_md)
    
    return {
        **state,
        "cover_letter_text": cover_letter_md.replace("```markdown", "").replace("```", "").strip(),
        "cover_letter_markdown": cover_letter_md,
        "messages": messages,
        "next_agent": "final_response",
        "current_task": "Finalizing documents"
    }

def save_cover_letter_to_markdown(content: str) -> str:
    """Saves cover letter to a markdown file with timestamp"""
    import datetime
    import os
    
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cover_letter_{timestamp}.md"
    filepath = os.path.join("outputs", filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filepath

def final_response_node(state: ResumeOptimizationState) -> ResumeOptimizationState:
    messages = state["messages"]
    old_ats_score = state.get("old_ats_score")
    new_ats_score = state.get("new_ats_score")
    analysis_report = state["analysis_report"]
    edited_resume = state["edited_resume_content"]
    
    cover_letter = state["cover_letter_text"]
    cover_letter_path = state.get("cover_letter_markdown", "").split("Saved to: ")[-1]

    messages.append(HumanMessage(content="Node: `final_response_node` - Generating final report."))

    # Save the optimized resume to a file
    saved_filepath = save_resume_to_markdown(edited_resume)
    messages.append(AIMessage(content=f"Optimized resume saved to: {saved_filepath}"))

    final_report_content = (
        f"--- Resume Optimization Report ---\n"
        f"**Original ATS Score:** {old_ats_score if old_ats_score is not None else 'N/A'}%\n"
        f"**Optimized ATS Score:** {new_ats_score if new_ats_score is not None else 'N/A'}%\n\n"
        f"--- Detailed Analysis of Original Resume ---\n"
        f"{analysis_report}\n\n"
        f"--- Optimized Resume Content ---\n"
        f"Saved to file: {saved_filepath}\n\n"
        f"```markdown\n{edited_resume}\n```\n\n"
        f"--- Professional Cover Letter ---\n"
        f"Saved to: {cover_letter_path}\n"
        f"{cover_letter}\n\n"
         f"--- Next Steps ---\n"
        "1. Review both documents\n"
        "2. Customize further if needed\n"
        "3. Submit with your application!"
    )

    messages.append(AIMessage(content=final_report_content))
    messages.append(SystemMessage(content="Node: `final_response_node` - Final report generated. Workflow complete."))
    return {
        **state,
        "messages": messages,
        "task_complete": True,
        "next_agent": "end",
        "current_task": "Completed",
        "saved_resume_path": saved_filepath  # Add the filepath to the state
    }
    
workflow = StateGraph(ResumeOptimizationState)

workflow.add_node("ingestion", ingestion_node)
workflow.add_node("keyword_extraction", keyword_extraction_node)
workflow.add_node("resume_analysis", resume_analysis_node)
workflow.add_node("human_review", human_review_node)
workflow.add_node("determine_next_step", determine_next_step) # This is now a router node
workflow.add_node("resume_editing", resume_editing_node)
workflow.add_node("final_ats_analysis", final_ats_analysis_node)
workflow.add_node("cover_letter_analysis", cover_letter_analysis_node)
workflow.add_node("cover_letter_generation", cover_letter_generation_node)
workflow.add_node("final_response", final_response_node)

workflow.set_entry_point("ingestion")

workflow.add_edge("ingestion", "keyword_extraction")
workflow.add_edge("keyword_extraction", "resume_analysis")
workflow.add_edge("resume_analysis", "human_review")


# KEY CHANGE: Conditional edges from the router node itself
workflow.add_conditional_edges(
    "human_review", # The node that returns the routing decision
    determine_next_step,   # The function that makes the routing decision
    {
        "resume_editing": "resume_editing", # Map the string "resume_editing" to the node "resume_editing"
        END: END                           # Map the END symbol to the graph's END
    }
)

workflow.add_edge("resume_editing", "final_ats_analysis")
workflow.add_edge("final_ats_analysis", "cover_letter_analysis")
workflow.add_edge("cover_letter_analysis", "cover_letter_generation")
workflow.add_edge("cover_letter_generation", "final_response")
workflow.add_edge("final_response", END)

final_workflow = workflow.compile(checkpointer=InMemorySaver())

# --- Dynamic User Input and Workflow Execution ---
print("--- ATS Resume Optimizer ---")

# 1. Get Job Description Input
jd_type = input("Do you want to provide a Job Description URL or paste text? (url/text): ").strip().lower()
job_description_raw = ""
if jd_type == "url":
    job_description_raw = input("Please enter the Job Description URL: ").strip()
elif jd_type == "text":
    print("Please paste the Job Description text (type 'END' on a new line when done):")
    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)
    job_description_raw = "\n".join(lines)
else:
    print("Invalid choice. Exiting.")
    exit()

# 2. Get Resume Input
print("\nPlease paste your Resume content:")
lines = []
while True:
    line = input()
    if line.strip().upper() == "END":
        break
    lines.append(line)
resume_raw_content = "\n".join(lines)

resume_format_choice = ""
while resume_format_choice not in ["markdown", "latex"]:
    resume_format_choice = input("Is your resume in 'markdown' or 'latex' format?: ").strip().lower()
    if resume_format_choice not in ["markdown", "latex"]:
        print("Invalid format. Please enter 'markdown' or 'latex'.")

# Construct initial state
initial_state = {
    "messages": [HumanMessage(content="Optimize my resume!")],
    "job_description_raw": job_description_raw,
    "resume_raw_content": resume_raw_content,
    "resume_format": resume_format_choice,
    "job_description_text": "",
    "resume_plain_text": "",
    "extracted_keywords": [],
    "analysis_report": "",
    "edited_resume_content": "",
    "human_feedback": "",
    "next_agent": "",
    "task_complete": False,
    "current_task": "",
    "old_ats_score": None,
    "new_ats_score": None
}

print("\n--- Running Workflow ---")
current_state = initial_state
thread_id = "user_session_123" # A fixed thread ID for console, replace with dynamic for multiple users

# Stream the graph, which now runs automatically through the 'human_review_node'
for s in final_workflow.stream(current_state, {"configurable": {"thread_id": thread_id}}):
    for key, value in s.items():
        if key != "__end__":
            # Print messages from the current step's state
            if "messages" in value and len(value["messages"]) > len(current_state["messages"]):
                new_messages = value["messages"][len(current_state["messages"]):]
                for msg in new_messages:
                    print(f"[{msg.type.upper()}] {msg.content}")
            
            print(f"Node Executed: {key}")
            current_state.update(value) # Update current_state with the latest step's output
        else:
            # Final state (contains "__end__")
            current_state.update(value)

# After the loop finishes (when END is reached)
print("\n--- Workflow Completed ---")
print(current_state["messages"][-1].content) # Final report message
            
print("\nWorkflow finished.")
if "saved_resume_path" in current_state:
    print(f"\nOptimized resume saved to: {current_state['saved_resume_path']}")