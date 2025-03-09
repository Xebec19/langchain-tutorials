import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API")

search_tool = SerperDevTool()

print("Welcome to Multi agent AI")
topic = input("Enter the topic: ")

llm = GoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=GOOGLE_AI_API_KEY)

researcher = Agent(
    role="Senior Researcher",
    goal=f"Uncover ground breaking technologies in {topic}",
    verbose=True,
    memory=True,
    backstory=(
    """
    Driven by curiousity, you're at the forefront of 
    innovation, eager to explore and share knowledge that could change
    the change    
    """
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=True
)

writer = Agent(
    role="Writer",
    goal=f"Narrate compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory=(
    """ 
        With a flair for simplifying complex topics, you craft
        engaging narratives that captivate and educate, bringing new
        discoveries to light in an accessible manner.
    """
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=False 
)

research_task = Task(
    description=(
        f"Identify the next big trend in {topic}."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points"
        "Its market opportunities, and potential risks."
    ),
    expected_output="A comprehensive 3 paragraphs long report on the topic",
    tools=[search_tool],
    agent=researcher
)

write_task = Task(
    description=(
        f"Compose an insightful article on {topic}"
        "Focus on the latest trends and how it's impacting the industry"
        "This article should be easy to understand, engaging, and positive"
    ),
    expected_output=f"A 4 paragraph article on {topic} fomatted as markdown",
    tools=[search_tool],
    agent=writer,
    async_execution=False,
    output_file="blog-post.md"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

result = crew.kickoff(inputs={"topic": "AI in healthcare"})
print(result)