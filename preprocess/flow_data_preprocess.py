from flowcontainer.extractor import extract
import binascii
import scapy.all as scapy


MAX_PACKET_NUMBER = 10  # 10
MAX_PACKET_LENGTH_IN_FLOW = 256
HEX_PACKET_START_INDEX = 0  # 48 # 76


def build_flow_data(pcap_file, flow_feature="flow bytes"):

    build_data = []

    if flow_feature == "flow bytes":

        # flow bytes feature
        build_data = []
        packets = scapy.rdpcap(pcap_file)

        hex_stream = []
        for i, packet in enumerate(packets):
            if i >= MAX_PACKET_NUMBER:
                break
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))

            packet_string = data.decode()

            # byte_list = re.findall(".{2}", packet_string)
            # packet_string = " ".join(byte_list)

            hex_stream.append(packet_string[HEX_PACKET_START_INDEX:min(len(packet_string), MAX_PACKET_LENGTH_IN_FLOW)])

        flow_data = "<pck>" + "<pck>".join(hex_stream)
        build_data.append(flow_data)

    elif flow_feature == "flow sequence":
        flows = extract(pcap_file,
                        filter='tcp or udp',
                        extension=["tcp.payload", "udp.payload"],
                        split_flag=False,
                        verbose=True)

        # flow sequence feature
        for key, flow in flows.items():
            flow_seq = []

            length_seq = flow.lengths
            for i, packet_length in enumerate(length_seq):
                if i >= MAX_PACKET_NUMBER:
                    break
                flow_seq.append(str(packet_length))

            flow_data = " ".join(flow_seq)
            build_data.append(flow_data)

    else:
        # payload bytes feature
        flows = extract(pcap_file,
                        filter='tcp or udp',
                        extension=["tcp.payload", "udp.payload"],
                        split_flag=False,
                        verbose=True)

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
