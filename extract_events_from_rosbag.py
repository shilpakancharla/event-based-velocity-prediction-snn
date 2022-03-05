import os
import sys
import rosbag
import shutil
import argparse
from os.path import basename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag") # Rosbag file to extract
    parser.add_argument("dest") # Destination for extracted data
    parser.add_argument("event_topic") # Event topic name
    
    args = parser.parse_args()

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    width, height = None, None
    event_sum = 0
    event_msg_sum = 0
    num_msgs_between_logs = 25
    output_name = os.path.basename(args.bag).split('.')[0] # /path/to/bag/out_hw1.bag ==> out_hw1
    path_to_events_file = os.path.join(args.dest, '{}.txt'.format(output_name))

    with open(path_to_events_file, 'w') as events_file:
        with rosbag.Bag(args.bag, 'r') as bag:
            # Look for topics available and save number of messages
            total_num_event_msgs = 0
            topics = bag.get_type_and_topic_info().topics
            for topic_name, topic_infor in topics.iteritems():
                total_num_event_msgs = topic_info.message_count
                print("{} has {} messages.".format(topic_name, topic_info.message_count))

            # Extract events to text file
            for topic, msg, t in bag.read_messages():
                if topic == args.event_topic:
                    if width is None:
                        width = msg.width
                        height = msg.height
                        print('Found sensor size: {} x {}'.format(width, height))
                        events_file.write("{} {}\n".format(width, height))

                    if event_msg_sum % num_msgs_between_logs == 0 or event_msg_sum >= total_num_event_msgs - 1:
                        print("Event messages: {} / {}".format(event_msg_sum + 1, total_num_event_msgs))
                    event_msg_sum += 1

                    for e in msg.events:
                        events_file.write(timestamp_str(e.ts) + " ")
                        events_file.write(str(e.x) + " ")
                        events_file.write(str(e.y) + " ")
                        events_file.write(("1 " if e.polarity else "0") + "\n")
                        event_sum += 1
            
        print("All events extracted.")
        print("Events: ", event_sum)