# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A chess play game implemented by group chat"""

from qwen_agent.agents import GroupChat
from qwen_agent.gui import WebUI
from qwen_agent.llm.schema import Message
from qwen_agent.tools.base import BaseTool, register_tool
import requests
import json

# Define a configuration file for a multi-agent:
# one real player, one NPC player, and one chessboard
NPC_NAME1 = '恶意软件检测专家'
NPC_NAME2 = '加密VPN检测专家'
USER_NAME = '网络安全主管'
CFGS = {
    'background':
        f'一个网络安全专家团队，{NPC_NAME1}是恶意软件检测专家，{NPC_NAME2}是加密VPN检测专家，针对用户提出的待测流量样本，需要团队内专家分析讨论得出最终检测结论。',
    'agents': [
        {
            'name':
                NPC_NAME1,
            'description':
                '负责执行恶意软件检测任务',
            'instructions':
                'Malware traffic detection service, input instructions about malware traffic detection tasks and network traffic data, return the answer of analysis results from TrafficLLM. Please note the traffic data must start with <packet>.',
            'selected_tools': ['trafficllm', 'code_interpreter'],
        },
        {
            'name':
                NPC_NAME2,
            'description':
                '负责执行加密VPN检测任务',
            'instructions':
                'Encrypted VPN traffic detection service, input instructions about encrypted VPN traffic detection tasks and network traffic data, return the answer of analysis results from TrafficLLM. Please note the question should be "Please conduct VPN Detection task" and the traffic data must start with <packet>.',
            'selected_tools': ['trafficllm', 'code_interpreter'],
        },
        {
            'name': USER_NAME,
            'description': '网络安全主管，负责分派网络安全威胁检测任务',
            'is_human': True
        },
    ],
}

llm_cfg = {
        # Use your own model service compatible with OpenAI API by vLLM/SGLang:
        'model': 'Qwen/Qwen3-30B-A3B-Thinking-2507',
        'model_server': 'http://localhost:8000/v1',  # api_base
        'api_key': 'EMPTY',

        'generate_cfg': {
            # When using vLLM/SGLang OAI API, pass the parameter of whether to enable thinking mode in this way
            'extra_body': {
                'chat_template_kwargs': {'enable_thinking': True}
            },

            # Add: When the content is `<think>this is the thought</think>this is the answer`
            # Do not add: When the response has been separated by reasoning_content and content
            # This parameter will affect the parsing strategy of tool call
            # 'thought_in_content': True,
        },
    }


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

def test(query: str = '<1,1>'):

    bot = GroupChat(agents=CFGS, llm=llm_cfg)  # {'model': 'qwen-max'}

    messages = [Message('user', query, name=USER_NAME)]
    for response in bot.run(messages=messages):
        print('bot response:', response)


def app_tui():
    # Define a group chat agent from the CFGS
    bot = GroupChat(agents=CFGS, llm=llm_cfg)  # {'model': 'qwen-max'}
    # Chat
    messages = []
    while True:
        query = input('user question: ')
        messages.append(Message('user', query, name=USER_NAME))
        response = []
        for response in bot.run(messages=messages):
            print('bot response:', response)
        messages.extend(response)


def app_gui():
    # Define a group chat agent from the CFGS
    bot = GroupChat(agents=CFGS, llm=llm_cfg)  # {'model': 'qwen-max'}
    chatbot_config = {
        'user.name': '网络安全主管',
        'prompt.suggestions': [
            'Please conduct network traffic detection task: <packet>: frame.encap_type: 1, frame.time: Jun 12, 2010 18:37:04.282112000 CST, frame.offset_shift: 0.000000000, frame.time_epoch: 1276339024.282112000, frame.time_delta: 0.001297000, frame.time_relative: 110.805986000, frame.number: 2184, frame.len: 60, frame.marked: 0, frame.protocols: eth:ethertype:ip:tcp, eth.dst: 00:e0:b1:87:f5:94, eth.dst_resolved: Alcatel-_87:f5:94, eth.src: 00:11:25:bb:ce:a1, eth.src_resolved: Ibm_bb:ce:a1, eth.type: 0x00000800, ip.version: 4, ip.hdr_len: 20, ip.dsfield: 0x00000000, ip.dsfield.dscp: 0, ip.dsfield.ecn: 0, ip.len: 40, ip.id: 0x000077d1, ip.flags: 0x00004000, ip.flags.rb: 0, ip.flags.df: 1, ip.flags.mf: 0, ip.frag_offset: 0, ip.ttl: 128, ip.proto: 6, ip.checksum: 0x00001429, ip.checksum.status: 2, ip.src: 192.168.1.102, ip.dst: 75.127.97.72, tcp.srcport: 2861, tcp.dstport: 80, tcp.stream: 0, tcp.len: 0, tcp.seq: 2123, tcp.nxtseq: 2123, tcp.ack: 1947143, tcp.hdr_len: 20, tcp.flags: 0x00000010, tcp.flags.res: 0, tcp.flags.ns: 0, tcp.flags.cwr: 0, tcp.flags.ecn: 0, tcp.flags.urg: 0, tcp.flags.ack: 1, tcp.flags.push: 0, tcp.flags.reset: 0, tcp.flags.syn: 0, tcp.flags.fin: 0, tcp.flags.str: \u00b7\u00b7\u00b7\u00b7\u00b7\u00b7\u00b7A\u00b7\u00b7\u00b7\u00b7, tcp.window_size: 11680, tcp.window_size_scalefactor: -2, tcp.checksum: 0x000033aa, tcp.checksum.status: 2, tcp.urgent_pointer: 0, tcp.time_relative: 110.805986000, tcp.time_delta: 0.001297000'
        ],
        'verbose': True
    }

    WebUI(
        bot,
        chatbot_config=chatbot_config,
    ).run()


if __name__ == '__main__':
    # test()
    # app_tui()
    app_gui()
