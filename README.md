# GauAI - Intelligent Data Analysis Framework

GauAI is a powerful framework for automated data analysis and machine learning, designed to turn data into actionable intelligence through a system of coordinated agents.

## Features

- **Intelligent Data Ingestion**: Automated CSV file loading with metadata extraction
- **Statistical Analysis**: Comprehensive statistical analysis with automated insights
- **ML Advisory**: Smart recommendations for machine learning approaches
- **Agent-based Architecture**: Coordinated system of specialized agents
- **Workflow Orchestration**: Automated execution of analysis workflows
- **Context Management**: Track analysis history and maintain context

## Project Structure

```
gauai/
├── core/
│   ├── data_management/    # Data ingestion and management
│   └── context/           # Analysis context and metadata
├── agents/
│   ├── ml_advisor/        # ML recommendations
│   ├── data_enrichment/   # Data enhancement suggestions
│   └── orchestrator/      # Workflow coordination
├── analysis/
│   ├── statistical/       # Statistical analysis
│   └── data_analysis/    # Data exploration and analysis
└── visualization/         # Data visualization components
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gauai.git
cd gauai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic example of using GauAI:

```python
from gauai.core.data_management import DataIngestionManager
from gauai.agents.orchestrator import OrchestratorAgent
from gauai.agents.ml_advisor import MLAdvisorAgent

# Initialize components
data_manager = DataIngestionManager()
ml_advisor = MLAdvisorAgent()
orchestrator = OrchestratorAgent()

# Register agents
orchestrator.register_agent("ml_advisor", ml_advisor)

# Load data
df = data_manager.load_csv("your_data.csv")

# Create and execute a workflow
workflow_config = {
    "steps": [
        {
            "name": "ml_analysis",
            "agent": "ml_advisor",
            "method": "analyze_dataset",
            "parameters": {"df": df}
        }
    ]
}

results = orchestrator.execute_workflow(workflow_config)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
