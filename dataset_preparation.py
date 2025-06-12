import os
import csv
import pandas as pd


input_folder = 'OriginalStoriesSeparated'
output_csv = 'dataset.csv'


labels = {
    "DoctorWho": 1,
    "XFiles": 2,
    "Stargate": 3,
    "StarTrek": 4,
    "Farscape": 5,
    "Babylon5": 6,
    "StarWarsRebels": 7,
    "Fringe": 8,
    "DoctorWhoSpinoffs": 9,
    "StarWarsBooks": 10,
    "Futurama": 0
}


def create_csv():
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['content', 'label'])

        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, mode='r', encoding='utf-8') as file:
                    lines = file.read().splitlines()
                    for line in lines:
                        if line.strip() and line.strip() != '<EOS>':
                            f_name = filename.replace('.txt', '')
                            writer.writerow([line, labels.get(f_name, 0)])


def load_dataset_and_modify():
    # load csv to pandas DataFrame
    df = pd.read_csv(output_csv, encoding='utf-8')

    df['name'] = df['name'].map(labels)


# print all txt files in the input folder
def list_txt_files():
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for file in txt_files:
        print(file)

# if __name__ == "__main__":
    # change_label()
