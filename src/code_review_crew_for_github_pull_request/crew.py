from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import ScrapeWebsiteTool
from .tools.github_tools import (
    FetchPRFilesTool,
    FindRelatedPRsTool,
    FindRelatedFilesTool,
)

@CrewBase
class CodeReviewCrewForGithubPullRequestCrew():
    """CodeReviewCrewForGithubPullRequest crew"""

    @before_kickoff
    def load_files_changed(self, inputs):
        self.files_changed = FetchPRFilesTool().run(inputs['pr_patch_url'])
        inputs['files_changed'] = self.files_changed
        return inputs

    @after_kickoff
    def add_disclaimer(self, inputs):
        inputs['raw'] = f"This review was made by a Crew.\n\n{inputs['raw']}"
        return inputs

    @agent
    def quick_review_assistant(self) -> Agent:
        return Agent(
            config=self.agents_config['quick_review_assistant'],
            tools=[
                ScrapeWebsiteTool(),
                FetchPRFilesTool(),
                FindRelatedPRsTool(),
                FindRelatedFilesTool(),
            ]
        )

    @agent
    def code_review_expert(self) -> Agent:
        return Agent(
            config=self.agents_config['code_review_expert']
        )

    @task
    def analyze_code_changes_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_code_changes_task'],
            tools=[]
        )

    @task
    def find_related_prs_task(self) -> Task:
        return Task(
            config=self.tasks_config['find_related_prs_task']
        )

    @task
    def analyze_related_files_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_related_files_task']
        )

    @task
    def generate_review_comment_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_review_comment_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CodeReviewCrewForGithubPullRequest crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
