from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
search_tool = SerperDevTool()

openrouter_key = os.getenv("OPENROUTER_API_KEY")
openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-6-luna")

def get_pipeline_llm():
    if openrouter_key:
        return LLM(model=f"openrouter/{openrouter_model}", api_key=openrouter_key)
    return None

@CrewBase
class Edu():
    """Crew for detecting misinformation in {topic} content"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def content_analyzer(self) -> Agent:
        llm = get_pipeline_llm()
        kwargs = {"config": self.agents_config['content_analyzer'], "verbose": True, "tools": [search_tool]}
        if llm:
            kwargs["llm"] = llm
        return Agent(**kwargs)

    @agent
    def fact_verifier(self) -> Agent:
        llm = get_pipeline_llm()
        kwargs = {"config": self.agents_config['fact_verifier'], "verbose": True, "tools": [search_tool]}
        if llm:
            kwargs["llm"] = llm
        return Agent(**kwargs)

    @agent
    def credibility_assessor(self) -> Agent:
        llm = get_pipeline_llm()
        kwargs = {"config": self.agents_config['credibility_assessor'], "verbose": True}
        if llm:
            kwargs["llm"] = llm
        return Agent(**kwargs)

    @task
    def claim_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['claim_extraction_task']
        )

    @task
    def claim_verification_task(self) -> Task:
        return Task(
            config=self.tasks_config['claim_verification_task']
        )

    @task
    def credibility_assessment_task(self) -> Task:
        return Task(
            config=self.tasks_config['credibility_assessment_task'],
            output_file='credibility_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Misinformation Credibility Assessment Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            max_retry_limit=2,
        )
