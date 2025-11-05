# Building Agents with TrafficLLM

## 1. Configure MCP Server

First, register TrafficLLM as a tool in the agent.

```
@register_tool('trafficllm')
class TrafficLLM(BaseTool):
    description = 'Network traffic detection service, input instructions about network traffic detection tasks and network traffic data, return the answer of analysis results from TrafficLLM. Please note the traffic data must start with <packet>.'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': 'Detailed description of the network traffic detection tasks and traffic data',
        'required': True,
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json.loads(params)['prompt']
        # prompt = urllib.parse.quote(prompt)
        print(prompt)
        human_instruction = prompt.split("<packet>")[0]
        print(human_instruction)
        traffic_data = "<packet>" + prompt.split("<packet>")[1]
        print(traffic_data)

        url = "http://localhost:8877/api/conversation"
        headers = {"Content-Type": "application/json"}
        data = {
            "human_instruction": human_instruction, "traffic_data": traffic_data,
        }
        response = requests.post(url, headers=headers, json=data)

        print(response.json())
        return response.json()
```

```
tools = [
        {
            'mcpServers': {  # You can specify the MCP configuration file
                'time': {
                    'command': 'uvx',
                    'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
                },
                'fetch': {
                    'command': 'uvx',
                    'args': ['mcp-server-fetch']
                }
            }
        },
        'security_qa',
        'trafficllm',
        'code_interpreter',  # Built-in tools
    ]
```

## 2. Deploy LLM and TrafficLLM Models

Deploy TrafficLLM using the following code.
```
python trafficllm_flask.py
```

Deploy a tool-calling LLM, such as [Qwen3](https://github.com/QwenLM/Qwen3).
```
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 --port 8000 --max-model-len 10000 --enable-reasoning --reasoning-parser deepseek_r1
```

## 3. Run Agent
Run agent to start the demo.
```
python assistant_qwen3.py
```
