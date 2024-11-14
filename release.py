import os
import subprocess
import re


def get_latest_tag():
    output = subprocess.check_output(['git', 'tag'])
    tags = output.decode('utf-8').split('\n')[:-1]
    latest_tag = sorted(tags, key=lambda t: tuple(map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', t).groups())))[-1]
    return latest_tag


def update_version_number(latest_tag, increment):
    major, minor, patch = map(int, re.match(r'v(\d+)\.(\d+)\.(\d+)', latest_tag).groups())
    if increment == 'X':
        major += 1
        minor, patch = 0, 0
    elif increment == 'Y':
        minor += 1
        patch = 0
    elif increment == 'Z':
        patch += 1
    new_version = f"v{major}.{minor}.{patch}"
    return new_version


def main():
    print("Latest version label：")
    latest_tag = get_latest_tag()
    print(latest_tag)

    print("Choose version number to be incremented：")
    increment = input().upper()

    while increment not in ['X', 'Y', 'Z']:
        print("Illegal input! Please input X/Y/Z")
        increment = input().upper()

    new_version = update_version_number(latest_tag, increment)
    print(f"New version number：{new_version}")

    print("Update version number and push？（y/n）")
    confirmation = input().lower()

    if confirmation == 'y':
        subprocess.run(['git', 'tag', new_version])
        subprocess.run(['git', 'push', 'origin', new_version])
        print("Version number created and pushed")
    else:
        print("Operaiton cancelled")


if __name__ == '__main__':
    main()
