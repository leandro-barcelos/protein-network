import os


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

if __name__ == "__main__":
  print(ROOT_DIR)