import os
import subprocess

metrics = ""
counter = 0


# java -jar ckjm_ext.jar projects/1_tulibee/tulibee.jar
def read_cmd_output(command):
    out = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')

    combine_output(out)


def find_jars(path):
    global metrics, counter
    rootdir = path
    jar_files = []
    for dir1 in os.listdir(rootdir):
        d = os.path.join(rootdir, dir1)
        if os.path.isdir(d):
            for file in os.listdir(d):
                file1 = os.path.join(d, file)
                if counter > 10:
                    break
                counter += 1

                if file.endswith((".jar")):
                    jar_files.append(file1)
                    run_metrics_tool(file1)
    write_file()


def run_metrics_tool(project):
    command = "java -jar ckjm_ext.jar " + project

    read_cmd_output(command)
    return command


def combine_output(out):
    global metrics
    metrics = metrics + out


def write_file():
    global metrics
    with open('Others/metrics.txt', 'w') as f:
        f.write(metrics)

    f.close()

    process_raw_metrics_from_cmd()


def process_raw_metrics_from_cmd():
    file = open('Others/metrics.txt', 'r')
    lines = file.readlines()

    processed = ""
    for line in lines:
        if not line.startswith(' ~') and line.strip() != "":
            processed = processed + line

    with open('Others/metrics.txt', 'w') as f:
        f.write(processed)

    f.close()
