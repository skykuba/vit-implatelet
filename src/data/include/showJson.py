import json

def print_tree(data, indent="", is_last=True):
    """Rekurencyjne drukowanie JSON jako struktura drzewa."""

    if isinstance(data, dict):
        keys = list(data.keys())
        for i, key in enumerate(keys):
            last = (i == len(keys) - 1)

            branch = "└── " if last else "├── "
            print(indent + branch + str(key))

            new_indent = indent + ("    " if last else "│   ")
            print_tree(data[key], new_indent, last)

    elif isinstance(data, list):
        clean_list = [x for x in data if x is not None]

        for i, item in enumerate(clean_list):
            last = (i == len(clean_list) - 1)

            #branch = "└── " if last else "├── "
            #print(indent + branch + str(item))


if __name__ == "__main__":
    with open("kegg_pathways.json", "r", encoding="utf-8") as f:
        kegg_data = json.load(f)

    print("KEGG Pathways\n")
    print_tree(kegg_data)