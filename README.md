# ARC-AGI Challenge 3 – Efficient Successor Learning

## Authors

- [Jeremy Gatineau](https://jeremygatineau.github.io/) 

## Overview
We are learning action representations through successor features, with various tricks to make learning as sample efficient as possible. Initial forked from [the StochasticGoose agent by Dries Smit](https://github.com/DriesSmit/ARC3-solution)

**Key Features:**
- WIP

## Setup Instructions

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Step 1: Clone Repository
```bash
git clone --recurse-submodules git@github.com:DriesSmit/ARC3-solution.git
cd ARC3-solution
```

### Step 2: Create Environment File
Copy the example environment file and set your API key (get your API key from [https://three.arcprize.org/user](https://three.arcprize.org/user)):
```bash
cd ARC-AGI-3-Agents
cp .env-example .env
# Then edit .env file and replace the empty ARC_API_KEY= with your actual API key
cd ..
```

### Step 3: Install Dependencies
```bash
make install
```

### Step 4: Configure Submodule
Add the following code to `ARC-AGI-3-Agents/agents/__init__.py` (under the imports and before `load_dotenv()`):

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_agent import *
```

Also, add the following field to the `FrameData` class in `ARC-AGI-3-Agents/agents/structs.py` (after the `full_reset` field):

```python
available_actions: list[GameAction] = Field(default_factory=list)
```

### Step 5: Run the Action Agent
```bash
make action
```

## Architecture
WIP

## Monitoring

The agent generates comprehensive logs and TensorBoard metrics:

```bash
# View training metrics
make tensorboard
# Open http://localhost:6006 in browser
```


## Files Structure

```
ARC3/
├── ARC-AGI-3-Agents/      # Competition framework (submodule)
├── custom_agents/
│   ├── __init__.py        # Agent registration
│   ├── action.py          # Main action learning agent
│   └── view_utils.py      # Visualization utilities
├── custom_agents.py       # Agent imports
├── Makefile               # Build commands
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── utils.py               # Shared utilities
```

## Additional Usage Examples

```bash
# Standard competition run
make action

# Run with specific game ID
uv run ARC-AGI-3-Agents/main.py --agent=action --game=vc33

# View logs and metrics
make tensorboard

# Clean generated files
make clean
```

