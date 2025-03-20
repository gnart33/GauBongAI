from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

class OrchestratorAgent:
    """Coordinates the execution of various analysis agents and tasks."""
    
    def __init__(self):
        self.agents = {}
        self.execution_history = []
        self.current_context = {}
        self.logger = logging.getLogger(__name__)
        
    def register_agent(self, name: str, agent: Any) -> None:
        """
        Register an agent with the orchestrator.
        
        Args:
            name: Name of the agent
            agent: Agent instance
        """
        self.agents[name] = agent
        self.logger.info(f"Registered agent: {name}")
        
    def execute_workflow(
        self,
        workflow_config: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Execute a workflow of analysis tasks.
        
        Args:
            workflow_config: Configuration defining the workflow steps
            context: Optional context data for the workflow
            
        Returns:
            Dict containing workflow results
        """
        self.current_context = context or {}
        results = {}
        
        for step in workflow_config["steps"]:
            try:
                step_result = self._execute_step(step)
                results[step["name"]] = step_result
                self._update_execution_history(step, step_result)
            except Exception as e:
                self.logger.error(f"Error executing step {step['name']}: {str(e)}")
                results[step["name"]] = {"error": str(e)}
                
        return results
    
    def _execute_step(self, step_config: Dict) -> Dict:
        """Execute a single workflow step."""
        agent_name = step_config["agent"]
        if agent_name not in self.agents:
            raise ValueError(f"Agent not found: {agent_name}")
            
        agent = self.agents[agent_name]
        method = getattr(agent, step_config["method"])
        
        return method(**step_config.get("parameters", {}))
    
    def _update_execution_history(
        self,
        step_config: Dict,
        result: Dict
    ) -> None:
        """Update the execution history with step results."""
        self.execution_history.append({
            "timestamp": datetime.now(),
            "step_name": step_config["name"],
            "agent": step_config["agent"],
            "method": step_config["method"],
            "parameters": step_config.get("parameters", {}),
            "status": "success" if "error" not in result else "error",
            "result_summary": self._summarize_result(result)
        })
        
    def _summarize_result(self, result: Dict) -> Dict:
        """Create a summary of step results for logging."""
        if "error" in result:
            return {"error": result["error"]}
            
        return {
            "output_keys": list(result.keys()),
            "output_types": {k: type(v).__name__ for k, v in result.items()}
        }
    
    def get_execution_history(
        self,
        step_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Retrieve execution history.
        
        Args:
            step_name: Optional step name to filter history
            
        Returns:
            List of execution history entries
        """
        if step_name:
            return [
                entry for entry in self.execution_history
                if entry["step_name"] == step_name
            ]
        return self.execution_history
    
    def get_agent(self, name: str) -> Any:
        """Retrieve a registered agent by name."""
        return self.agents.get(name) 