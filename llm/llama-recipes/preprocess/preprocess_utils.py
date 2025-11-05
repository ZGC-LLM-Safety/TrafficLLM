from flow_data_preprocess import build_flow_data
from packet_data_preprocess import build_packet_data
import random
import json
import os


MAX_SAMPLING_NUMBER = 100  # 5000 # number of samples per class
TRAINING_SAMPLE_RATIO = 0.95


def split_dataset(build_data, sampling=True):
    random.shuffle(build_data)
    if sampling is True:
        train_nb = int(min(MAX_SAMPLING_NUMBER, len(build_data)) * TRAINING_SAMPLE_RATIO)
        test_nb = int(min(MAX_SAMPLING_NUMBER, len(build_data)) * (1 - TRAINING_SAMPLE_RATIO))
    else:
        train_nb = int(len(build_data) * TRAINING_SAMPLE_RATIO)
        test_nb = int(len(build_data) * (1 - TRAINING_SAMPLE_RATIO))
    train_data = build_data[:train_nb]
    test_data = build_data[train_nb:train_nb + test_nb]

    return train_data, test_data


def write_dataset(dataset, output_path):
    random.shuffle(dataset)
    with open(output_path, "w", encoding="utf-8") as fin:
        for data in dataset:
            json.dump(data, fin)
            fin.write("\n")
        # json.dump(dataset, fin, indent=4, separators=(',', ': '))


def write_labels(labels, output_path):
    label_dict = {}
    for i, label in enumerate(labels):
        label_dict[label] = i
    with open(output_path, "w", encoding="utf-8") as fin:
        json.dump(label_dict, fin, indent=4, separators=(',', ': '))


def build_dataset(args, path, file):
    build_data = []
    files_path = os.path.join(path, file)
    pcaps = os.listdir(files_path)
    for pcap in pcaps:
        if args.granularity == "flow":
            pcap_data = build_flow_data(os.path.join(files_path, pcap))
        else:
            pcap_data = build_packet_data(os.path.join(files_path, pcap))
        build_data.extend(pcap_data)

    train_data, test_data = split_dataset(build_data)
    return train_data, test_data


def save_dataset(args, train_dataset, test_dataset):
    write_dataset(train_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_" +
                                              args.granularity + "_train.json"))
    write_dataset(test_dataset, os.path.join(args.output_path, args.output_name + "_" + args.traffic_task + "_" +
                                             args.granularity + "_test.json"))


def build_td_text_dataset(traffic_data, first_label=None, second_label=None, task_name=None, granularity=None):
    """Building the text datasets of traffic detection task"""
    if task_name == "EMD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the ENCRYPTED MALWARE DETECTION TASK to determine " \
                      "which application category the encrypted beign or malicious traffic belongs to. The categories " \
                      "include 'BitTorrent, FTP, Facetime, Gmail, MySQL, Outlook, SMB, Skype, Weibo, WorldOfWarcraft," \
                      "Cridex, Geodo, Htbot, Miuref, Neris, Nsis-ay, Shifu, Tinba, Virut, Zeus'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted malware detection task: "
        #
        # output = "This might be a " + first_label + \
        #          " traffic " + granularity + ". The category is likely to be recognized as " + second_label + "."

    elif task_name == "EAC":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the ENCRYPTED APP CLASSIFICATION TASK to determine " \
                      "which APP category the encrypted traffic belongs to. "
        # The categories " \
        #                       "include '163Mail, 51cto, Acm, Adobe, Alibaba, Alicdn, Alipay, Amap, AmazonAWS, AmpProject, Apple," \
        #                       "Arxiv, Asus, Atlassian, AzureEdge, Baidu, Bilibili, Biligame, Booking, LA'." \

        output = second_label
        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted App classification task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "BND":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                       "traffic features, and payloads. Please conduct the BOTNET DETECTION TASK to determine " \
                       "which type of network the traffic belongs to. The categories " \
                       "include 'IRC, Neris, RBot, Virut, normal'."

        output = second_label
        # instruction = "Below is a traffic " + granularity + ". Please conduct the botnet detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "EVD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the ENCRYPTED VPN DETECTION TASK to determine " \
                      "which behavior or application category the VPN encrypted traffic belongs to. The categories " \
                      "include 'aim, bittorrent, email, facebook, ftps, hangout, icq, netflix, sftp, skype, spotify, " \
                      "vimeo, voipbuster, youtube'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the encrypted VPN detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "MDD":
        instruction = "Below is a traffic " + granularity + ". Please conduct the malicious DoH detection task: "

        output = "The traffic category is likely to be recognized as " + second_label + "."

    elif task_name == "TBD":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                      "traffic features, and payloads. Please conduct the TOR BEHAVIOR DETECTION TASK to determine " \
                      "which behavior or application category the traffic belongs to under the Tor network. " \
                      "The categories include 'audio, browsing, chat, file, mail, p2p, video, voip'."

        output = second_label

    elif task_name == "APT":
        instruction = "Given the following traffic data <" + granularity + "> that contains protocol fields, " \
                                                                           "traffic features, and payloads. Please conduct the APT DETECTION TASK to determine " \
                                                                           "which behavior or application category the traffic belongs to under the APT attacks. " \
                                                                           "The categories include 'APT and normal'."

        output = second_label

        # instruction = "Below is a traffic " + granularity + ". Please conduct the Tor behavior detection task: "
        #
        # output = "The traffic category is likely to be recognized as " + second_label + "."

    # elif task_name == "ATD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the adware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."
    #
    # elif task_name == "RTD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the ransomware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."
    #
    # elif task_name == "STD":
    #     instruction = "Below is a traffic " + granularity + ". Please conduct the scareware traffic detection task: "
    #
    #     output = "The traffic category is likely to be recognized as " + second_label + "."

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction + "\n<" + granularity + ">: " + data,
                "output": output
            }
        )

    return dataset


def build_tg_text_dataset(traffic_data, traffic_label, granularity=None):
    """Building the text datasets of traffic generation task"""
    instruction = "Please generate a " + granularity + " of " + traffic_label + " traffic."

    dataset = []
    for data in traffic_data:
        dataset.append(
            {
                "instruction": instruction,
                "output": data
            }
        )

    return dataset


def build_tu_text_dataset(traffic_data, fields=None):
    """Building the text datasets of traffic understanding task"""

    knowledge_fields = []
    api_calls = []

    if "IP" in fields:
        knowledge_fields.extend(
            ["IP Version", "IP Header Length", "Differentiated Services Field", "Total Length",
             "Identification", "IP Flags", "Fragment Offset", "Time to Live", "Protocol", "IP Header Checksum",
             "Source Address", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-IP-version", "scapy-IP-ihl", "scapy-IP-tos", "scapy-IP-len", "scapy-IP-id", "scapy-IP-flags",
             "scapy-IP-frag", "scapy-IP-ttl", "scapy-IP-proto", "scapy-IP-chksum", "scapy-IP-src", "scapy-IP-dst"]
        )

    if "TCP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "Sequence Number", "Acknowledge Number",
             "TCP Flags", "Window", "TCP Header Checksum", "Urgent Pointer", "Destination Address"]
        )
        api_calls.extend(
            ["scapy-TCP-sport", "scapy-TCP-dport", "scapy-TCP-seq", "scapy-TCP-ack", "scapy-TCP-flags",
             "scapy-TCP-window", "scapy-TCP-chksum", "scapy-TCP-urgptr", "scapy-TCP-options"]
        )

    if "UDP" in fields:
        knowledge_fields.extend(
            ["Source Port", "Destination Port", "UDP Length", "UDP Header Checksum"]
        )
        api_calls.extend(
            ["scapy-UDP-sport", "scapy-UDP-dport", "scapy-UDP-len", "scapy-UDP-chksum"]
        )

    if "TLS" in fields:
        knowledge_fields.extend(
            ["Content Type", "Record Version", "TLS Message", "Message Type", "Handshake Version",
             "Cipher Suites", "Extensions"]
        )
        api_calls.extend(
            ["scapy-TLS-type", "scapy-TLS-version", "scapy-TLS-msg", "scapy-TLS-msg-msgtype", "scapy-TLS-msg-version",
             "scapy-TLS-msg-ciphers", "scapy-TLS-msg-ext"]
        )

    if "DNS" in fields:
        knowledge_fields.extend(
            ["Transaction ID", "Response", "Opcode", "Authoritative", "Truncated", "Recursion Desired",
             "Recursion Available", "Z", "Answer Authenticated", "Non-Authenticated", "Questions", "Answer RRs",
             "Authority RRs", "Additional RRs", "Queries", "Answers"]
        )
        api_calls.extend(
            ["scapy-DNS-id", "scapy-DNS-qr", "scapy-DNS-opcode", "scapy-DNS-aa", "scapy-DNS-tc", "scapy-DNS-rd",
             "scapy-DNS-ra", "scapy-DNS-z", "scapy-DNS-ad", "scapy-DNS-cd", "scapy-DNS-qdcount", "scapy-DNS-ancount",
             "scapy-DNS-nscount", "scapy-DNS-arcount", "scapy-DNS-qd", "scapy-DNS-an"]
        )

    if "http.HTTPRequest" in fields:
        knowledge_fields.extend(
            ["Headers", "Host", "User-Agent", "Accept", "Connection", "Method", "Path", "Http-Version", "Range",
             "Accept-Language", "Additional-Headers"]
        )
        api_calls.extend(
            ["scapy-http.HTTPRequest-Headers", "scapy-http.HTTPRequest-Host", "scapy-http.HTTPRequest-User-Agent",
             "scapy-http.HTTPRequest-Accept", "scapy-http.HTTPRequest-Connection", "scapy-http.HTTPRequest-Method",
             "scapy-http.HTTPRequest-Path", "scapy-http.HTTPRequest-Http-Version", "scapy-http.HTTPRequest-Range",
             "scapy-http.HTTPRequest-Accept-Language", "scapy-http.HTTPRequest-Additional-Headers"]
        )

    if "http.HTTPResponse" in fields:
        knowledge_fields.extend(
            ["Headers", 'Accept-Ranges', 'Server', 'Cache-Control', 'Connection', 'Date', 'Content-Length',
             'Content-Range', 'Content-Type', 'Last-Modified', 'Additional-Headers', 'Status-Line']
        )
        api_calls.extend(
            ["scapy-http.HTTPResponse-Headers", "scapy-http.HTTPResponse-Accept-Ranges",
             "scapy-http.HTTPResponse-Server", "scapy-http.HTTPResponse-Cache-Control",
             "scapy-http.HTTPResponse-Connection", "scapy-http.HTTPResponse-Date",
             "scapy-http.HTTPResponse-Content-Length", "scapy-http.HTTPResponse-Content-Range",
             "scapy-http.HTTPResponse-Content-Type", "scapy-http.HTTPResponse-Last-Modified",
             "scapy-http.HTTPResponse-Additional-Headers", "scapy-http.HTTPResponse-Status-Line"]
        )

    if "GeoIP" in fields:
        knowledge_fields.extend(
            ["source address", "destination address"]
        )
        api_calls.extend(
            ["<geoip-src>", "<geoip-dst>"]
        )

    if "JA3" in fields:
        knowledge_fields.extend(
            ["client fingerprints", "server fingerprints"]
        )
        api_calls.extend(
            ["<ja3-client>", "<ja3-server>"]
        )

    dataset = []

    for data in traffic_data:
        index = random.randint(0, len(knowledge_fields) - 1)
        if "GeoIP" in fields or "JA3" in fields:
            dataset.append(
                {
                    "instruction": "Please analyze the " + knowledge_fields[index] + " in the packet: " + data,
                    "output":  "<" + api_calls[index] + ">"
                }
            )
        else:
            dataset.append(
                {
                    "instruction": "What is " + knowledge_fields[index] + " in the packet: " + data,
                    "output": "<" + api_calls[index] + ">"
                }
            )

    return dataset
