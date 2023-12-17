# CS438 Final Project

## Table of Contents

- [Getting Started - Method 1](#getting-started---method-1)
  - [Prerequisites](#prerequisites)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Activate the Conda Environment](#activate-the-conda-environment)
  - [Launch Jupyter Notebook](#launch-jupyter-notebook)
  - [Run the Notebook](#run-the-notebook)
- [Getting Started - Method 2](#getting-started---method-2)

## Getting Started - Method 1

### Local Jupyter Notebook setup instructions for the CS438 Final Project.

These instructions will guide you through setting up the environment and running the Jupyter notebook for the CS438 Final Project.

### Prerequisites

Before you begin, ensure you have the following installed:
- Anaconda or Miniconda

### Setting Up the Environment

1. **Clone the Project Repository**

   If you have `git` installed, you can clone the repository using the following command:

    ```bash
    git clone https://github.com/RyanZurrin/CS438-Final-Project.git
    ```


Alternatively, you can download the ZIP file of the project and extract it to your desired location.

2. **Create a Conda Environment**

Navigate to the directory containing the `CS438.yml` environment file, then create a new Conda environment by running:

```bash
conda env create -f CS438.yml
```

This command will create a new environment named CS438 and install all the necessary dependencies.

3. **Activate the Conda Environment**

To activate the environment, run:

```bash
conda activate CS438
```

4. **Launch Jupyter Notebook**

To launch Jupyter Notebook, run:

```bash
jupyter notebook
```

This will open a new tab in your browser. From here, you can navigate to the `CS438-Final-Project` directory and open the `CS438-Final-Project.ipynb` notebook.

5. **Run the Notebook**

With the notebook open, you can run each cell individually by clicking the `Run` button in the toolbar or by pressing `Shift + Enter`. Alternatively, you can run all cells at once by clicking `Kernel` > `Restart & Run All` in the toolbar.

## Getting Started - Method 2

### Google Colab setup instructions for the CS438 Final Project.

If you do not have Anaconda or Miniconda installed, you can run the notebook in Google Colab.

1. **Open Google Colab**

   Navigate to https://colab.research.google.com/.
2. **Upload the Notebook**

   In the left sidebar, Click `File` > `Upload notebook`, then select the `CS438-Final-Project.ipynb` notebook from the project directory.
3. **Run the Notebook**

   With the notebook open, you can run each cell individually by clicking the `Play` button to the left of the cell or by pressing `Shift + Enter`. Alternatively, you can run all cells at once by clicking `Runtime` > `Run all` in the toolbar.
4. **Save the Notebook**

   To save your changes, click `File` > `Save a copy in Drive` in the toolbar.

Remember that you need access to the internet to use Google Colab, and the session will disconnect after 90 minutes of inactivity.