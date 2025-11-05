from flowcontainer.extractor import extract


MAX_PACKET_NUMBER = 10
MAX_PACKET_LENGTH_IN_FLOW = 256


def build_flow_data(pcap_file):
    flows = extract(pcap_file,
                    filter='ip',
                    extension=["tcp.payload", "udp.payload"],
                    split_flag=False,
                    verbose=True)

    build_data = []

    for key, flow in flows.items():
        if len(flow.extension.values()) == 0:
            continue
        packet_list = list(flow.extension.values())[0]
        hex_stream = []
        for i, packet in enumerate(packet_list):
            if i >= MAX_PACKET_NUMBER:
                break
            hex_stream.append(packet[0][:min(len(packet[0]), MAX_PACKET_LENGTH_IN_FLOW)])
        flow_data = "<pck>" + "<pck>".join(hex_stream)
        # print(flow_data)
        build_data.append(flow_data)

    return build_data
