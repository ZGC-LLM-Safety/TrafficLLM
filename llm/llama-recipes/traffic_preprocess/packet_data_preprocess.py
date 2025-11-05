from flowcontainer.extractor import extract


MAX_PACKET_LENGTH = 1024


def build_packet_data(pcap_file):
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
        for i, packet in enumerate(packet_list):
            build_data.append(packet[0][:min(len(packet[0]), MAX_PACKET_LENGTH)])

    return build_data
