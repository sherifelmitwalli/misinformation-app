from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
from dotenv import load_dotenv
from crewai_tools import SerperDevTool

# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
search_tool = SerperDevTool()

@CrewBase
class Edu():
    """Crew for detecting misinformation in {topic} content"""

    # Updated configuration file paths
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def content_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['content_analyzer'],
            verbose=True,
            tools=[search_tool]
        )

    @agent
    def fact_verifier(self) -> Agent:
        return Agent(
            config=self.agents_config['fact_verifier'],
            verbose=True,
            tools=[search_tool]
        )

    @agent
    def credibility_assessor(self) -> Agent:
        return Agent(
            config=self.agents_config['credibility_assessor'],
            verbose=True
        )

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
        """Creates the Misinformation Detection Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
