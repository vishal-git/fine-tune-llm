{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8e1b1fd3-6e78-49f7-a42d-4663561c4d1b",
   "metadata": {},
   "source": [
    "This notebook contains a step-by-step guide for fine-tuning `gpt-3.5-turbo` model. We will use the Titanic dataset for this tutorial. We will first build a (simple) classification model using the target label `survived`. Then we will use Shapley analysis to identify features that contributed to each passenger's survival estimate. The ouput from Shapley analysis will then be used to fine-tune an LLM model. We would like to fine-tune a GPT model in such a way that when we provide the output data points from the Shapley analysis to the LLM model, it should be able to generate a summary in plain English that explains why the passenger did or did not survive. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11fd2cd7-7a3e-4bec-9b37-caefc9495b6a",
   "metadata": {},
   "source": [
    "Here are the steps we will follow:\n",
    "\n",
    "1. Read the Titanic dataset and prepare the data for modeling.\n",
    "2. Build a simple classification model.\n",
    "3. Perform Shapley analysis to identify the top contributors for each passanger.\n",
    "4. Prepare training data for fine-tuning a GPT model (by converting Shapley output into plain English).\n",
    "5. Fine-tune `gpt-3.5-turbo` using the training data prepared in the previous step.\n",
    "6. Make inference on a new passenger (not in the training data) to make sure that the fine-tuned model is working as expected.\n",
    "7. Build a Gradio UI where a user can choose a passanger and view the results."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3435587c-9ff2-4c1c-8401-5ddd9f2e71e9",
   "metadata": {},
   "source": [
    "### import packages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "162211a5-6b9f-49d4-a7a9-654a043b868d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# data processing\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "# modeling\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "import joblib\n",
    "# shapley analysis\n",
    "import shap\n",
    "# utilities\n",
    "from pprint import pprint\n",
    "import json\n",
    "import os\n",
    "import sys\n",
    "# gradio UI\n",
    "import gradio as gr\n",
    "import matplotlib.pyplot as plt\n",
    "import tempfile\n",
    "# openai\n",
    "from openai import OpenAI\n",
    "from dotenv import find_dotenv, load_dotenv"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "636ed0ed-482f-47eb-bf98-c6244e0a97aa",
   "metadata": {},
   "source": [
    "### read data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "efe0479e-5cab-498d-90ba-5210ed65d5ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Ticket Class</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Siblings</th>\n",
       "      <th>Family Members</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Braund, Mr. Owen Harris</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</th>\n",
       "      <td>1</td>\n",
       "      <td>Upper</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>Cherbourg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Heikkinen, Miss. Laina</th>\n",
       "      <td>1</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Futrelle, Mrs. Jacques Heath (Lily May Peel)</th>\n",
       "      <td>1</td>\n",
       "      <td>Upper</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Allen, Mr. William Henry</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Moran, Mr. James</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8.4583</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Queenstown</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>McCarthy, Mr. Timothy J</th>\n",
       "      <td>0</td>\n",
       "      <td>Upper</td>\n",
       "      <td>male</td>\n",
       "      <td>54.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>51.8625</td>\n",
       "      <td>E46</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Palsson, Master. Gosta Leonard</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>3</td>\n",
       "      <td>2</td>\n",
       "      <td>21.0750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</th>\n",
       "      <td>1</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>11.1333</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Nasser, Mrs. Nicholas (Adele Achem)</th>\n",
       "      <td>1</td>\n",
       "      <td>Middle</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>30.0708</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Cherbourg</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sandstrom, Miss. Marguerite Rut</th>\n",
       "      <td>1</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>16.7000</td>\n",
       "      <td>G6</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Bonnell, Miss. Elizabeth</th>\n",
       "      <td>1</td>\n",
       "      <td>Upper</td>\n",
       "      <td>female</td>\n",
       "      <td>58.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>26.5500</td>\n",
       "      <td>C103</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Saundercock, Mr. William Henry</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>20.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Andersson, Mr. Anders Johan</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>39.0</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>31.2750</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Vestrom, Miss. Hulda Amanda Adolfina</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>14.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7.8542</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Hewlett, Mrs. (Mary D Kingcome)</th>\n",
       "      <td>1</td>\n",
       "      <td>Middle</td>\n",
       "      <td>female</td>\n",
       "      <td>55.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>16.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Rice, Master. Eugene</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>male</td>\n",
       "      <td>2.0</td>\n",
       "      <td>4</td>\n",
       "      <td>2</td>\n",
       "      <td>29.1250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Queenstown</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Williams, Mr. Charles Eugene</th>\n",
       "      <td>1</td>\n",
       "      <td>Middle</td>\n",
       "      <td>male</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>13.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Vander Planke, Mrs. Julius (Emelia Maria Vandemoortele)</th>\n",
       "      <td>0</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>31.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>18.0000</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Southampton</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Masselmani, Mrs. Fatima</th>\n",
       "      <td>1</td>\n",
       "      <td>Lower</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7.2250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Cherbourg</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    Survived Ticket Class  \\\n",
       "Name                                                                        \n",
       "Braund, Mr. Owen Harris                                    0        Lower   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...         1        Upper   \n",
       "Heikkinen, Miss. Laina                                     1        Lower   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)               1        Upper   \n",
       "Allen, Mr. William Henry                                   0        Lower   \n",
       "Moran, Mr. James                                           0        Lower   \n",
       "McCarthy, Mr. Timothy J                                    0        Upper   \n",
       "Palsson, Master. Gosta Leonard                             0        Lower   \n",
       "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)          1        Lower   \n",
       "Nasser, Mrs. Nicholas (Adele Achem)                        1       Middle   \n",
       "Sandstrom, Miss. Marguerite Rut                            1        Lower   \n",
       "Bonnell, Miss. Elizabeth                                   1        Upper   \n",
       "Saundercock, Mr. William Henry                             0        Lower   \n",
       "Andersson, Mr. Anders Johan                                0        Lower   \n",
       "Vestrom, Miss. Hulda Amanda Adolfina                       0        Lower   \n",
       "Hewlett, Mrs. (Mary D Kingcome)                            1       Middle   \n",
       "Rice, Master. Eugene                                       0        Lower   \n",
       "Williams, Mr. Charles Eugene                               1       Middle   \n",
       "Vander Planke, Mrs. Julius (Emelia Maria Vandem...         0        Lower   \n",
       "Masselmani, Mrs. Fatima                                    1        Lower   \n",
       "\n",
       "                                                       Sex   Age  Siblings  \\\n",
       "Name                                                                         \n",
       "Braund, Mr. Owen Harris                               male  22.0         1   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...  female  38.0         1   \n",
       "Heikkinen, Miss. Laina                              female  26.0         0   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)        female  35.0         1   \n",
       "Allen, Mr. William Henry                              male  35.0         0   \n",
       "Moran, Mr. James                                      male   NaN         0   \n",
       "McCarthy, Mr. Timothy J                               male  54.0         0   \n",
       "Palsson, Master. Gosta Leonard                        male   2.0         3   \n",
       "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)   female  27.0         0   \n",
       "Nasser, Mrs. Nicholas (Adele Achem)                 female  14.0         1   \n",
       "Sandstrom, Miss. Marguerite Rut                     female   4.0         1   \n",
       "Bonnell, Miss. Elizabeth                            female  58.0         0   \n",
       "Saundercock, Mr. William Henry                        male  20.0         0   \n",
       "Andersson, Mr. Anders Johan                           male  39.0         1   \n",
       "Vestrom, Miss. Hulda Amanda Adolfina                female  14.0         0   \n",
       "Hewlett, Mrs. (Mary D Kingcome)                     female  55.0         0   \n",
       "Rice, Master. Eugene                                  male   2.0         4   \n",
       "Williams, Mr. Charles Eugene                          male   NaN         0   \n",
       "Vander Planke, Mrs. Julius (Emelia Maria Vandem...  female  31.0         1   \n",
       "Masselmani, Mrs. Fatima                             female   NaN         0   \n",
       "\n",
       "                                                    Family Members     Fare  \\\n",
       "Name                                                                          \n",
       "Braund, Mr. Owen Harris                                          1   7.2500   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...               1  71.2833   \n",
       "Heikkinen, Miss. Laina                                           1   7.9250   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                     1  53.1000   \n",
       "Allen, Mr. William Henry                                         1   8.0500   \n",
       "Moran, Mr. James                                                 1   8.4583   \n",
       "McCarthy, Mr. Timothy J                                          1  51.8625   \n",
       "Palsson, Master. Gosta Leonard                                   2  21.0750   \n",
       "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)                3  11.1333   \n",
       "Nasser, Mrs. Nicholas (Adele Achem)                              1  30.0708   \n",
       "Sandstrom, Miss. Marguerite Rut                                  2  16.7000   \n",
       "Bonnell, Miss. Elizabeth                                         1  26.5500   \n",
       "Saundercock, Mr. William Henry                                   1   8.0500   \n",
       "Andersson, Mr. Anders Johan                                      6  31.2750   \n",
       "Vestrom, Miss. Hulda Amanda Adolfina                             1   7.8542   \n",
       "Hewlett, Mrs. (Mary D Kingcome)                                  1  16.0000   \n",
       "Rice, Master. Eugene                                             2  29.1250   \n",
       "Williams, Mr. Charles Eugene                                     1  13.0000   \n",
       "Vander Planke, Mrs. Julius (Emelia Maria Vandem...               1  18.0000   \n",
       "Masselmani, Mrs. Fatima                                          1   7.2250   \n",
       "\n",
       "                                                   Cabin     Embarked  \n",
       "Name                                                                   \n",
       "Braund, Mr. Owen Harris                              NaN  Southampton  \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...   C85    Cherbourg  \n",
       "Heikkinen, Miss. Laina                               NaN  Southampton  \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)        C123  Southampton  \n",
       "Allen, Mr. William Henry                             NaN  Southampton  \n",
       "Moran, Mr. James                                     NaN   Queenstown  \n",
       "McCarthy, Mr. Timothy J                              E46  Southampton  \n",
       "Palsson, Master. Gosta Leonard                       NaN  Southampton  \n",
       "Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)    NaN  Southampton  \n",
       "Nasser, Mrs. Nicholas (Adele Achem)                  NaN    Cherbourg  \n",
       "Sandstrom, Miss. Marguerite Rut                       G6  Southampton  \n",
       "Bonnell, Miss. Elizabeth                            C103  Southampton  \n",
       "Saundercock, Mr. William Henry                       NaN  Southampton  \n",
       "Andersson, Mr. Anders Johan                          NaN  Southampton  \n",
       "Vestrom, Miss. Hulda Amanda Adolfina                 NaN  Southampton  \n",
       "Hewlett, Mrs. (Mary D Kingcome)                      NaN  Southampton  \n",
       "Rice, Master. Eugene                                 NaN   Queenstown  \n",
       "Williams, Mr. Charles Eugene                         NaN  Southampton  \n",
       "Vander Planke, Mrs. Julius (Emelia Maria Vandem...   NaN  Southampton  \n",
       "Masselmani, Mrs. Fatima                              NaN    Cherbourg  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('../data/titanic.csv', index_col='Name')\n",
    "\n",
    "# let's drop unnecessary columns and rename a few\n",
    "df = df.drop(columns=['PassengerId', 'Ticket'])\\\n",
    "    .rename(columns={'Pclass': 'Ticket Class', 'Parch': 'Family Members', 'SibSp': 'Siblings'})\n",
    "\n",
    "# convert the ticket class from numeric to string\n",
    "classes = {1: 'Upper', 2: 'Middle', 3: 'Lower'}\n",
    "df['Ticket Class'] = df['Ticket Class'].map(classes)\n",
    "\n",
    "# number of people the passenger was traveling with (0 if traveling alone)\n",
    "df['Family Members'] = 1 + df['Family Members']\n",
    "\n",
    "# change the column type to categorical \n",
    "cat_cols = ['Ticket Class', 'Sex', 'Embarked']\n",
    "\n",
    "for cat_col in cat_cols:\n",
    "    df[cat_col] = df[cat_col].astype('category')\n",
    "\n",
    "# make values for `embarked` more interpretable \n",
    "ports = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}\n",
    "df['Embarked'] = df['Embarked'].map(ports)\n",
    "\n",
    "df.head(20)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b27905be-e85f-4443-abe5-16d2a8c350ae",
   "metadata": {},
   "source": [
    "Are Cabin and Pclass the same thing?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "041b9c61-2aa5-4723-8efa-7ff674c7aef0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th>Ticket Class</th>\n",
       "      <th>Lower</th>\n",
       "      <th>Middle</th>\n",
       "      <th>Upper</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cabin</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>False</th>\n",
       "      <td>12</td>\n",
       "      <td>16</td>\n",
       "      <td>176</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>True</th>\n",
       "      <td>479</td>\n",
       "      <td>168</td>\n",
       "      <td>40</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Ticket Class  Lower  Middle  Upper\n",
       "Cabin                             \n",
       "False            12      16    176\n",
       "True            479     168     40"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cabin_assigned = pd.isnull(df['Cabin'])\n",
    "\n",
    "pd.crosstab(cabin_assigned, df['Ticket Class'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d51278c6-b303-4011-84db-93a3a7270739",
   "metadata": {},
   "source": [
    "No. So we should use `Cabin`. Let's use a binary flag for cabin-assigned, and drop `Cabin`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0383a725-db7c-4f7f-8fa2-3557daa80bb4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Cabin\n",
       "True     687\n",
       "False    204\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Cabin'] = df['Cabin'].isna()\n",
    "df['Cabin'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b76eaed-0cc3-416a-bb09-2a2edae7cc99",
   "metadata": {},
   "source": [
    "Let's create dummies for the rest of the categorical features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "b4146f91-2608-4c6c-b54f-450f94cbd622",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Ticket Class: Middle</th>\n",
       "      <th>Ticket Class: Upper</th>\n",
       "      <th>Sex: male</th>\n",
       "      <th>Embarked: Queenstown</th>\n",
       "      <th>Embarked: Southampton</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Age</th>\n",
       "      <th>Siblings</th>\n",
       "      <th>Family Members</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Name</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Braund, Mr. Owen Harris</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Heikkinen, Miss. Laina</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Futrelle, Mrs. Jacques Heath (Lily May Peel)</th>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Allen, Mr. William Henry</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    Ticket Class: Middle  \\\n",
       "Name                                                                       \n",
       "Braund, Mr. Owen Harris                                            False   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...                 False   \n",
       "Heikkinen, Miss. Laina                                             False   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                       False   \n",
       "Allen, Mr. William Henry                                           False   \n",
       "\n",
       "                                                    Ticket Class: Upper  \\\n",
       "Name                                                                      \n",
       "Braund, Mr. Owen Harris                                           False   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...                 True   \n",
       "Heikkinen, Miss. Laina                                            False   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                       True   \n",
       "Allen, Mr. William Henry                                          False   \n",
       "\n",
       "                                                    Sex: male  \\\n",
       "Name                                                            \n",
       "Braund, Mr. Owen Harris                                  True   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...      False   \n",
       "Heikkinen, Miss. Laina                                  False   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)            False   \n",
       "Allen, Mr. William Henry                                 True   \n",
       "\n",
       "                                                    Embarked: Queenstown  \\\n",
       "Name                                                                       \n",
       "Braund, Mr. Owen Harris                                            False   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...                 False   \n",
       "Heikkinen, Miss. Laina                                             False   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                       False   \n",
       "Allen, Mr. William Henry                                           False   \n",
       "\n",
       "                                                    Embarked: Southampton  \\\n",
       "Name                                                                        \n",
       "Braund, Mr. Owen Harris                                              True   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...                  False   \n",
       "Heikkinen, Miss. Laina                                               True   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                         True   \n",
       "Allen, Mr. William Henry                                             True   \n",
       "\n",
       "                                                    Survived   Age  Siblings  \\\n",
       "Name                                                                           \n",
       "Braund, Mr. Owen Harris                                    0  22.0         1   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...         1  38.0         1   \n",
       "Heikkinen, Miss. Laina                                     1  26.0         0   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)               1  35.0         1   \n",
       "Allen, Mr. William Henry                                   0  35.0         0   \n",
       "\n",
       "                                                    Family Members     Fare  \\\n",
       "Name                                                                          \n",
       "Braund, Mr. Owen Harris                                          1   7.2500   \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...               1  71.2833   \n",
       "Heikkinen, Miss. Laina                                           1   7.9250   \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)                     1  53.1000   \n",
       "Allen, Mr. William Henry                                         1   8.0500   \n",
       "\n",
       "                                                    Cabin  \n",
       "Name                                                       \n",
       "Braund, Mr. Owen Harris                              True  \n",
       "Cumings, Mrs. John Bradley (Florence Briggs Tha...  False  \n",
       "Heikkinen, Miss. Laina                               True  \n",
       "Futrelle, Mrs. Jacques Heath (Lily May Peel)        False  \n",
       "Allen, Mr. William Henry                             True  "
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_dummies = pd.get_dummies(df[cat_cols], drop_first=True)\n",
    "\n",
    "df = df_dummies.merge(df.drop(columns=cat_cols), left_index=True, right_index=True)\n",
    "\n",
    "# convert underscores to colons to improve legibility\n",
    "df.columns = [x.replace('_', ': ') for x in df.columns]\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2ae38e2-3e29-4d33-83ce-c65fef62c2bb",
   "metadata": {},
   "source": [
    "Are there any missing values?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "169a96f6-15cf-4fb4-8086-04fa49486ea1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Age    177\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "miss_ct = df.isnull().sum()\n",
    "miss_ct[miss_ct != 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bbec5aa4-f63b-4407-ab62-4b54f53683b3",
   "metadata": {},
   "source": [
    "`Age` has some missing values; let's just replace them with median values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "769077eb-8a3c-4bbd-af8f-1904b11b097d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ticket Class: Middle     0\n",
       "Ticket Class: Upper      0\n",
       "Sex: male                0\n",
       "Embarked: Queenstown     0\n",
       "Embarked: Southampton    0\n",
       "Survived                 0\n",
       "Age                      0\n",
       "Siblings                 0\n",
       "Family Members           0\n",
       "Fare                     0\n",
       "Cabin                    0\n",
       "age_missing              0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['age_missing'] = df['Age'].isna()\n",
    "df['Age'] = df['Age'].fillna(df['Age'].median())\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ca52705b-36c4-401a-8fd5-1a30d874e0be",
   "metadata": {},
   "source": [
    "No more missing values!"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "26195edb-a177-4683-9362-7eaea0df5bc0",
   "metadata": {},
   "source": [
    "### build a model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ee392349-0d63-4652-b3cf-3c84c23686ac",
   "metadata": {},
   "source": [
    "Since the goal of this exercise is not to build the best model, but focus on how we can fine-tune a GPT model, we will simply train the (classification) model on the entire dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "82582e61-c11c-46ad-8d9f-beb047ebdc64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((891, 11), (891,))"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = df['Survived']\n",
    "X = df.drop(columns=['Survived'])\n",
    "\n",
    "X.shape, y.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31042211-81e3-4b5d-8f3d-e3517a06f447",
   "metadata": {},
   "source": [
    "Let's skip hyper-parameter optimization and use cherry-picked values instead."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c53f4b8-2709-4c82-8d4b-c07754d38054",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf = RandomForestClassifier(n_estimators=100, min_samples_split=0.03, random_state=314)\n",
    "clf.fit(X, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41321eab-29f1-4377-bea6-4fe0cea9870f",
   "metadata": {},
   "source": [
    "Save the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "25c104f3-130d-42ec-b407-68853cbf1981",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['../models/titanic_model_v1.json']"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "joblib.dump(clf, '../models/titanic_model_v1.json')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e583f4a6-600a-432f-a259-5a898c1ee2de",
   "metadata": {},
   "source": [
    "### Shapley analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e7d54b30-4e4d-4a6d-9e0c-baf93ff9f399",
   "metadata": {},
   "outputs": [],
   "source": [
    "explainer = shap.TreeExplainer(clf)\n",
    "shap_values = explainer(X)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6f06854f-80cf-4151-9e18-c0f119f9d2a9",
   "metadata": {},
   "source": [
    "Let's take a random passanger."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cc7b47aa-9fb6-4ed2-b9b6-84a6248ee196",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "578\n"
     ]
    }
   ],
   "source": [
    "random_index = np.random.randint(0, len(X))\n",
    "print(random_index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "2b09173e-35a7-4a2a-944e-7cb94250a2b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "random_index = 526 # note: the index number is different than above cuz I ran this multiple times"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "abe38c7e-eb20-42d2-ac87-a0e7cd4f608f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'survived'"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "random_person = X.iloc[random_index]\n",
    "surv = 'did not Survive' if y.iloc[random_index] == 0 else 'survived' \n",
    "surv"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8c8e0280-5a02-41b2-8b0c-9a12e651514f",
   "metadata": {},
   "source": [
    "Let's grab the Shapley values for this passanger and view the waterfall chart."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "4e27ceba-8d83-48d3-a57f-d88248437b75",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA68AAAJYCAYAAACNeND8AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAADc40lEQVR4nOzdd1xT5/4H8M9J2HtvEVEEB4p7b62jtVNt1arVqvX2tra11p+1w/bW6q22t8suRx3V2jrrwln3VpwoCoIoyF6yCUnO74+UaEiAgMBB83m/XnlpnvOc53xPRJJvniWIoiiCiIiIiIiIqAGTSR0AERERERERUVWYvBIREREREVGDx+SViIiIiIiIGjwmr0RERERERNTgMXklIiIiIiKiBo/JKxERERERETV4TF6JiIiIiIiowWPySkRERERERA0ek1ciIiIiIiJq8Ji8EhERERERUYPH5JWIiIiIiIgaPCavRERERERE/1Cr1Zg3bx6aNm0Kc3NzNG3aFAsXLkRISAjUanW12/v555/h7++PkpKSOojWtAiiKIpSB0FERERERNQQLF68GNOnT8e7776LNm3awNHRERMnTsSXX36JiRMnVru94uJiBAQEYM6cOZg+fXodRGyYSqXCuXPnEBMTg5KSEri4uKBTp07w8/Or9LxDhw4hOjq6wuNjx46Fra1trdczBpNXIiIiIiKif3To0AFubm7Ys2cPAOCbb77B3LlzkZqaCisrqxq1+X//93/4888/cevWLQiCUJvhVujvv/9GXFwcQkND4ejoiOjoaKSlpWH48OHw8vKq8LzU1FTk5ubqlImiiGPHjsHe3h4jR46sk3rGMDO6JhERERER0WOsuLgYly5dwqeffqotW7FiBZ5++ukaJ64AMGrUKCxcuBAHDx5E//79ayPUSqWlpSE2NhZdunRB27ZtAQBBQUHYuHEjTp8+jWeeeabCcz09PeHp6alTlpKSAqVSiWbNmtVZPWNwzisREREREZm8V199FdbW1lCpVPjwww8hCAK8vb1x+fJlDBw4UK/+3bt3YWVlhUmTJumU79+/H+bm5njnnXe0ZR06dICLiwu2bt1a5/cBAHFxcRAEAS1atNCWmZmZITg4GKmpqcjPz69Wezdv3gSAKpPN2q5XHpNXIiIiIiIyeWPHjsVrr70GAPj222/x22+/Ydq0aQCA9u3b69X39fXF5MmTsWbNGty+fRsAcP36dYwcORJDhw7FV199pVO/ffv2OH78eJVxqNVqFBcXG/WoaAZoZmYmHB0dYWFhoVPu4eGhPW4stVqN2NhYeHp6wt7evt7qGcJhw0REREREZPL69++Pv//+G7a2tnjjjTcgk8nw0UcfAQCaNGli8Jz3338fy5YtwxdffIHPPvsMTz31FAICArBu3TrIZLr9hIGBgfjtt9+qjCMlJQU7duwwKubRo0cbTAALCwthY2OjV15WVlBQYFT7AJCQkICSkhIEBQXVaz1DmLwSEREREREBuHz5Mlq1aqVNPDMzM2FmZgY7OzuD9X19fTFlyhQsXboU58+fR1FREQ4fPmxw9VxnZ2cUFRVVmFiWcXV1xbBhw4yK19ra2mC5UqmEXC7XKy8rU6lURrUPaIb4ymQyBAYG1ms9Q5i8EhERERERAbh06RIGDx5crXNmzpyJxYsX4/Llyzh69Ch8fX0N1isb4lvVasOWlpZVbmdTFTMzM4MJalmZocTWkNLSUty+fRt+fn6VLlhV2/UqwuSViIiIiIhMXk5ODhISEhAaGqotc3V1hVKpRF5eXoXzMz///HMAmt5OFxeXCtvPzs6GjY1Nhb2lZVQqFUpKSoyK2crKSm94MqAZHmxoaHBhYSEAGL2vanx8vFGrAtd2vYoweSUiIiIiIpN3+fJlAECbNm20ZSEhIQCAW7du6ZSXWbRoEZYtW4bFixfjvffew+eff45ly5YZbP/WrVs6q/9WJDU19aHnvLq6uiIpKQkKhUJn0aa0tDTtcWPcvHkT5ubmCAgIqNd6FWHySkREREREJu/SpUsAdJPXbt26AQDOnTunl7z+9ddfmD17Nj777DP8+9//RkxMDH788Ud88MEHBhd4On/+PMaOHVtlHLUx5zUwMBCXL19GVFSUdp9XlUqFGzduwMPDQzuHV6lUIj8/H1ZWVnrDeIuKipCYmIhmzZrBzKzitLG261WGySsREREREZm8y5cvw9fXV2fob2BgIFq3bo39+/fr7OcaERGBsWPHYuzYsfjggw8AALNmzcLPP/9ssPc1IiICWVlZeOaZZ6qMozbmvHp4eCAwMBBnzpxBUVERHB0dER0djby8PPTp00dbLy0tDTt27ED79u3RsWNHnTZiY2MhimKVQ3xru15luM8rERERERGZvMuXLxscGjxp0iRs374dRUVFAIDExEQMHz4c7dq1w9KlS7X1fHx8MGnSJKxevRq3bt3SaWPDhg3w9/dH//796/YmHtC3b1+EhoYiJiYGJ06cgFqtxpAhQ+Dt7W3U+Tdv3oS1tXWFC1DVVb3KCGJFO9sSERERERGZuHv37iEwMBALFy7Eq6++Wu3zS0pKEBAQgNmzZ+Ott96qgwhNB3teiYiIiIiIKuDo6IhZs2Zh0aJFUKvV1T5/xYoVMDc3x7Rp0+ogOtPCnlciIiIiIiJq8NjzSkRERERERA0ek1ciIiIiIiJq8Ji8EhERERERUYPH5JWIiIiIiIgaPCavRERERERE1OAxeSUiIiIiIqqG3Nxc9O3bF7m5uVKHYlKYvBIREREREVVDbm4uDh8+zOS1njF5JSIiIiIiogaPySsRERERERE1eExeiYiIiIiIqMFj8kpERERERFQNDg4O6NatGxwcHKQOxaQIoiiKUgdBREREREREVBn2vBIREREREVXTlStXpA7B5DB5JSIiIiIiqiaVSiV1CCaHySsREREREVE1ubi4SB2CyWHySkREREREVE3Ozs5Sh2BymLwSERERERFVU2xsrNQhmBwmr0RERERERNTgcascIiIiIiKiarp37x4cHR2lDsOksOeViIiIiIiomvLy8qQOweQweSUiIiIiIqqmjIwMqUMwOUxeiYiIiIiIqkkQBKlDMDmc80pEREREREQNHnteiYiIiIiIqunq1atSh2BymLwSERERERFVU2lpqdQhmBwmr0RERERERNXk5OQkdQgmh8krERERERFRNbm7u0sdgslh8kpERERERFRNMTExUodgcpi8EhERERERUYPH5JWIiIiIiKiaGjduLHUIJofJKxERERERUTUVFRVJHYLJYfJKRERERERUTWlpaVKHYHLMpA6AiIiIHvDZemDvRamjICIiO2tg2/uAOVOmhkIQRVGUOggiIiL6h9UooEQpdRRERAQAf8wAXuxp8JBarYZMxoGs9YmvNhERERERUXkyAfjf9goP37hxox6DIYDJKxERERERkT61CJyJAc7dNHi4pKSkngMiJq9ERERERESGmMmA73YaPOTg4FDPwRCTVyIiIiIiIkOUamDdMSA1R++Qt7d3/cdj4pi8EhERERERVUStBn7Zq1fMOa/1j8krERERERFRRdQi8P1OQFEqdSQmj8krERERERFRZTLy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yWHySkREjywrK92hvcObCvC1e/h2LSy4TQ4REVFDw+SViIgeWc2aNdN5LpcJeLuD7KHf3EoViodsgYiIHndt27aVOgSTw+SViIgeWZGRkXplr4YKsJA/XLsceUxERFWJioqSOgSTw+SViIgeK85WAia0frhtc2Qyvj0SEVHlFBylU+/47kxERI8sLy8vg+XT28mgfIju0w6etbRhLBERPbYcHBykDsHkcJ9XIiJ6LOUUizUe/ht77RI6tg+rzXCIiOgxU1hYCBsbG6nDMClmUgdARERUF5ysat57asZxSUREVIXo6GiEhYVJHYZJ4dszERFROQEBAVKHQEREROUweSUiIiqnoKBA6hCIiKiB8/f3lzoEk8PklYiIqJz09HSpQyAiogaupKRE6hBMDpNXIiIiIiKiakpNTZU6BJPD5JWIiKictm3bSh0CERERlcOtcoiIiMq5du0aWrZsafCYSi1CoTK+LWtz7hlLRPQ4UqlUkMvlUodhUrhVDhERUTkKhaLCY3OOqrHwrPHf+64aKsP4VhzoRET0uImJiUFISIjUYZgUvpsSERGV4+DgYLC8sFTEz5eMT1zlAnAolisXExE9joqLi6UOweQweSUiIirHy8vLYPnaKBG5FXfK6pEJQGFhYS1FRUREDYm9vb3UIZgcJq9ERETlREdH65WJooj/nVOjujNYra2taycoIiJqUHx9faUOweQweSUiIjLCwQQR17OA6q5yKDfj8hJERI+j69evSx2CyWHySkREVI6/v79e2Tfn1JDXYOHg/Ly8WoiIiIiI+HUwERFROSUlJTrPb+WI2BFX/V5XIiJ6fPn5+UkdgslhzysREVE5qampOs8XX1BDVsPtWp2cnB4+ICIianBUqmps+k21gskrERFRJfIVIpZcFqGqYbcrVxsmIno8JScnSx2CyWHySkREVE5oaKj2779dE1FQWvO2FIpq7K1DREREFWLySkREVE5MTAyA+9vjPMxcV5mMb7VERI+jVq1aSR2CyeE7KhERUTnFxcUAgP23RdzMebi2XFxcHj4gIiJqcG7duiV1CCaHqw0TEVG9ySkWMeuIGltiRBSWAp29BXzVV4b2nsathhSVKeKdg2ocuyvCQg48GSjgf31lcLe5f/71TBG/RqqxN15EbA5gZwG09xDwaQ8ZOnoZdx17e3sAwNcRmu1xajrfFQAyMjIA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C6ampmrDe3NzczXaLi4uxrVr12BoaKhab9fa2hpxcXFIT09X6zW9ceMGJElSW/tV2zaJiEh/MHklIqJGITExEUlJSapEszouX74MQDN57dGjBwDg/PnzatuEEHjuuecQGRmJY8eOqYa8vvLKK/i///s/LF++HMHBwRrHuXjxIiZMmFDtOB9WG/e8Ojo6wtvbG2fPnkVubi6sra0RFRWF+/fvIzAwUFUvKSkJe/fuRefOndGlSxdV+cmTJ1FQUAAXFxdYWFggJycH169fR3p6Orp3766a4blDhw6IjY3Fnj174O/vDxMTE9y+fRuxsbFo1aqV2vBkbdskIiL9weSViIgahdLEMzk5GZs2bdLY3qFDB43hwGWFhobCzc1NrQcQKOmZbNu2LQ4fPowXX3xRVf7BBx9g9+7dCA4OVlu+xcHBARMnTsTGjRvxySefqA27vXDhAtLS0jBy5MhqnWdd6dOnDywtLXHt2jUUFBTAzs4OgwcPhouLS5X7ent7IzIyEuHh4cjLy4OxsTHs7e3x2GOPwcvLS1XPxcUFI0eOxIULF/DPP/8gPz8fVlZW6Nq1q6rHV9c2iYhIfzB5JSKiRqF0yO+6deuwbt06je0//PCDVslr2V7XUi+++CLmz5+vGs76888/46OPPsL06dMxbdo0jfqvv/461q5di1WrVuHdd99VlW/fvh0eHh7o16+fLqdX5xQKBbp3717psGtXV9dyz9XHxwc+Pj5aHcfR0RFDhgypsp4ubRIRkX6QhDazIxARETUikydPxtGjR3Hx4kUoFAqtZq3NyMiAt7c3lixZgilTplTruPn5+fDy8sLbb7+N1157TW1bXl4esrKysGTJEixduhTJycmwt7ev1nGIiIgaI8358omIiPRAbGwsHBwc0Lt3b63qW1tbY968eVi6dKlqdlxdrVu3DkZGRpg+fbrGtlWrVsHBwQFLly6tVttERESNHXteiYhI74SHhyMhIQEAYGlpWaMZimtLbGwsIiMjVc8DAwM5KREREdFDmLwSERERERFRg8dhw0RERERERNTgMXklIiIiIiKiBo/JKxERERERETV4TF6JiIiIiIiowWPySkRERERERA0ek1ciIiIiIiJq8Ji8EhERERERUYPH5JWIiIiIiIgaPCavRERERERE1OD9P4XoS/96sDOGAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x650 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "_shap_values = shap_values[random_index, :, 1]\n",
    "shap.plots.waterfall(_shap_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "269bac35-6d45-4fd7-a1c1-6d4e82391af5",
   "metadata": {},
   "source": [
    "We will grab overall means for all features and append them to the Shapley dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "fcc2210c-bfbf-4534-9c3d-488a90177e40",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>overall_avg</th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Ticket Class: Middle</th>\n",
       "      <td>0.206510</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket Class: Upper</th>\n",
       "      <td>0.242424</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Sex: male</th>\n",
       "      <td>0.647587</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked: Queenstown</th>\n",
       "      <td>0.086420</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Embarked: Southampton</th>\n",
       "      <td>0.722783</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                       overall_avg  value\n",
       "Ticket Class: Middle      0.206510   True\n",
       "Ticket Class: Upper       0.242424  False\n",
       "Sex: male                 0.647587  False\n",
       "Embarked: Queenstown      0.086420  False\n",
       "Embarked: Southampton     0.722783   True"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "all_means = pd.DataFrame(X.mean())\n",
    "all_means.columns = ['overall_avg']\n",
    "\n",
    "all_means['value'] = random_person.values\n",
    "all_means.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e1acb08d-2c37-4647-9a28-2ed160d9056c",
   "metadata": {},
   "source": [
    "Create a consolidated Shapley dataset with all Shapley values and the overall means."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "d92f287a-40c6-45a9-bbe5-7950c7ed0957",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>contribution</th>\n",
       "      <th>overall_avg</th>\n",
       "      <th>value</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Sex: male</th>\n",
       "      <td>0.368614</td>\n",
       "      <td>0.647587</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Ticket Class: Middle</th>\n",
       "      <td>0.099736</td>\n",
       "      <td>0.206510</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Siblings</th>\n",
       "      <td>0.018092</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age_missing</th>\n",
       "      <td>0.002236</td>\n",
       "      <td>0.198653</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Family Members</th>\n",
       "      <td>0.000879</td>\n",
       "      <td>1.381594</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      contribution  overall_avg  value\n",
       "Sex: male                 0.368614     0.647587  False\n",
       "Ticket Class: Middle      0.099736     0.206510   True\n",
       "Siblings                  0.018092     0.523008      0\n",
       "age_missing               0.002236     0.198653  False\n",
       "Family Members            0.000879     1.381594      1"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_shap = pd.DataFrame(_shap_values.values, index=X.columns)\n",
    "df_shap.columns = ['contribution']\n",
    "\n",
    "df_shap = df_shap.merge(all_means, left_index=True, right_index=True)\\\n",
    "    .sort_values('contribution', ascending=False)\n",
    "\n",
    "df_shap.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8e43752-a580-4389-89a5-d3ee0561a1ec",
   "metadata": {},
   "source": [
    "### prepare data for LLM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dbebd504-dc79-4137-9f91-c712fdf8786a",
   "metadata": {},
   "source": [
    "#### Sample 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "287bd7f2-cc4d-4a4e-8572-35efcdc97649",
   "metadata": {},
   "source": [
    "Now we need to create some explanations based on the Shapley output. These explanations will become part of the data that will be used to fine-tune a GPT model. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63101794-4398-4682-85d6-4c7941d5bb0e",
   "metadata": {},
   "source": [
    "But first let's organize the Shapley output into text format. For simplicity (and to save some time), let's just use the top three contributors from the Shapley output and create convert that tabular data into plain text format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 117,
   "id": "5c374b40-2686-4da6-9249-f009bb4c7fa8",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_n = 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "31382ed0-9aed-41d7-8514-9da302ad0821",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Feature --> Sex: male = False, Contribution: 36.9%, Overall Average: 64.8%\\n'\n",
      " 'Feature --> Ticket Class: Middle = True, Contribution: 10.0%, Overall '\n",
      " 'Average: 20.7%\\n'\n",
      " 'Feature --> Siblings = 0, Contribution: 1.8%, Overall Average: 52.3%\\n')\n"
     ]
    }
   ],
   "source": [
    "sample_1 = ''\n",
    "\n",
    "for i, row in df_shap.head(top_n).iterrows():\n",
    "    if i == 'Fare':\n",
    "        sample_1 += f\"Feature --> {i} = ${row['value']:.0f}, Contribution: {row['contribution']:.1%}, \"\n",
    "    elif row['value'] in [True, False, 1, 0]:\n",
    "        sample_1 += f\"Feature --> {i} = {row['value']}, Contribution: {row['contribution']:.1%}, \"\n",
    "    else:\n",
    "        sample_1 += f\"Feature --> {i} = {row['value']:.1%}, Contribution: {row['contribution']:.1%}, \"\n",
    "    \n",
    "    if row['value'] in [True, False, 1, 0]:\n",
    "        sample_1 += f\"Overall Average: {row['overall_avg']:.1%}\\n\"\n",
    "    else:\n",
    "        if i == 'Fare':\n",
    "            sample_1 += f\"Overall Average: ${row['overall_avg']:.1f}\\n\"\n",
    "        else:\n",
    "            sample_1 += f\"Overall Average: {row['overall_avg']:.1f}\\n\"\n",
    "\n",
    "pprint(sample_1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "868e0dad-e254-47ae-a302-be215ac74210",
   "metadata": {},
   "source": [
    "We will also append the outcome, aka whether the passanger survived the Titanic crash or not."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "6a6592bc-c541-4e19-a834-51ebcf5b752c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Feature --> Sex: male = False, Contribution: 36.9%, Overall Average: 64.8%\\n'\n",
      " 'Feature --> Ticket Class: Middle = True, Contribution: 10.0%, Overall '\n",
      " 'Average: 20.7%\\n'\n",
      " 'Feature --> Siblings = 0, Contribution: 1.8%, Overall Average: 52.3%\\n'\n",
      " '\\n'\n",
      " 'Outcome: This passenger survived.\\n'\n",
      " '\\n')\n"
     ]
    }
   ],
   "source": [
    "sample_1 += f'\\nOutcome: This passenger {surv}.\\n\\n'\n",
    "pprint(sample_1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2a29e4b6-7ad5-4609-b052-a84af22ed15d",
   "metadata": {},
   "source": [
    "Based on these data points, let's manually generate some explanation for why this passager survived."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "60b8d9bc-b1ac-4784-b588-e52b55324b6e",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_1 = \"The largest positive contributor to why this passanger survived was her gender. The fact that this passanger was a female improved her survival by as much as 36.6%. The second largest contributor was her ticket class: 'Middle', which improved her chances of survival by 10.0%. She was among the 20.7% of all passengers who were in the 'Middle' ticket class. Unlike the majority of the passengers (52.3%), she did not have any siblings, which improved her survival changes just slightly by 1.8%.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "71f3c7bc-5420-4cdf-9859-429aa8e55e33",
   "metadata": {},
   "source": [
    "OpenAI fine-tuning API requires at least ten samples, so we will have to do this nine more time. Let's create a function."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6a86534d-b533-422f-a897-f81b541140b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_sample(i=None, plot=True):\n",
    "    if i == None:\n",
    "        random_index = np.random.randint(0, len(X))\n",
    "        print(random_index)\n",
    "    else:\n",
    "        random_index = i\n",
    "        \n",
    "    random_person = X.iloc[random_index]\n",
    "    surv = 'did not Survive' if y.iloc[random_index] == 0 else 'survived' \n",
    "\n",
    "    _shap_values = shap_values[random_index, :, 1]\n",
    "    if plot == True:\n",
    "        shap.plots.waterfall(_shap_values)\n",
    "\n",
    "    all_means['value'] = random_person.values\n",
    "\n",
    "    df_shap = pd.DataFrame(_shap_values.values, index=X.columns)\n",
    "    df_shap.columns = ['contribution']\n",
    "    df_shap['abs_contribution'] = abs(df_shap['contribution'])\n",
    "    df_shap = df_shap.merge(all_means, left_index=True, right_index=True)\n",
    "    df_shap = df_shap.sort_values(by='abs_contribution', ascending=False)\n",
    "    print(df_shap.head(3))\n",
    "\n",
    "    _sample = ''\n",
    "\n",
    "    for i, row in df_shap.head(3).iterrows():\n",
    "        if i == 'Fare':\n",
    "            _sample += f\"Feature --> {i} = ${row['value']:.0f}, Contribution: {row['contribution']:.1%}, \"\n",
    "        elif (row['value'] in [True, False, 1, 0]) | (i == 'Siblings'):\n",
    "            _sample += f\"Feature --> {i} = {row['value']}, Contribution: {row['contribution']:.1%}, \"\n",
    "        else:\n",
    "            _sample += f\"Feature --> {i} = {row['value']:.1%}, Contribution: {row['contribution']:.1%}, \"\n",
    "        \n",
    "        if row['value'] in [True, False, 1, 0]:\n",
    "            _sample += f\"Overall Average: {row['overall_avg']:.1%}\\n\"\n",
    "        else:\n",
    "            if i == 'Fare':\n",
    "                _sample += f\"Overall Average: ${row['overall_avg']:.1f}\\n\"\n",
    "            else:\n",
    "                _sample += f\"Overall Average: {row['overall_avg']:.1f}\\n\"\n",
    "    \n",
    "    _sample += f'\\nOutcome: This passenger {surv}.\\n\\n'\n",
    "\n",
    "    return _sample"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e27b70e-9d19-4c12-99b2-56170ff89e81",
   "metadata": {},
   "source": [
    "We will use this function to create nine more sample data points."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bf9862f8-4f6b-45c2-8495-6082245652b2",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "31d09d65-09d1-4972-ab0b-ef64b8d3bc32",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg  value\n",
      "Sex: male               -0.200476          0.200476     0.647587   True\n",
      "Cabin                   -0.036367          0.036367     0.771044   True\n",
      "Ticket Class: Upper     -0.024778          0.024778     0.242424  False\n"
     ]
    }
   ],
   "source": [
    "sample_2 = get_sample(800, False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "8dc7f515-e5fa-4fa3-9bfe-96a23bdcf120",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_2 = \"Unfortunately, this passenger did not survive, mainly driven by the fact that he was a male, which reduced his chance of survival by -20.0%. He stayed in a Cabin, and his ticket class was 'Upper', which further reduced his changes by -3.6% and -2.5% respectively.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "fd38c669-121d-4df3-b6e7-e5b60a775ea0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "('Feature --> Sex: male = True, Contribution: -20.0%, Overall Average: 64.8%\\n'\n",
      " 'Feature --> Cabin = True, Contribution: -3.6%, Overall Average: 77.1%\\n'\n",
      " 'Feature --> Ticket Class: Upper = False, Contribution: -2.5%, Overall '\n",
      " 'Average: 24.2%\\n'\n",
      " '\\n'\n",
      " 'Outcome: This passenger did not Survive.\\n'\n",
      " '\\n')\n"
     ]
    }
   ],
   "source": [
    "pprint(sample_2)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a31bde7f-7d2d-4050-9dda-82ed98e9f6fa",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "cabd7c90-5567-4f50-996e-383c372ab60b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA6sAAAJYCAYAAACEk3CGAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy80BEi2AAAACXBIWXMAAA9hAAAPYQGoP6dpAADtOElEQVR4nOzdd3gU5doG8Ht203vvhCSQQgqEXqVjEEFRioKIoCIcRY6gcrA31COoqIdPUECKdANIFZAuIC2UEBJISAIhjXTSy+7O90fMwrIpm5DNJOH+XVeuZN95Z+aZzWazz7xNEEVRBBEREREREVEzIpM6ACIiIiIiIqL7MVklIiIiIiKiZofJKhERERERETU7TFaJiIiIiIio2WGySkRERERERM0Ok1UiIiIiIiJqdpisEhERERERUbPDZJWIiIiIiIiaHSarRERERERE1OwwWSUiIiIiIqJmh8kqERERERERNTtMVomIiIiIqNlRqVSYP38+2rVrB0NDQ7Rr1w4LFixAQEAAVCpVvY+3dOlSeHp6oqysTA/Rkj4IoiiKUgdBRERERER0r8WLF2PWrFl488030bFjR1hbW2Pq1Kn4+uuvMXXq1Hofr7S0FF5eXnj33Xcxa9YsPURcPaVSiXPnziEuLg5lZWWws7ND9+7d4eHhUee+OTk5iIiIQFZWFoqLi2FgYABbW1t06tQJbdu21ah7584dnD17Frdv30ZpaSksLCzQvn17dOrUCQYGBvq6PL1iskpERERERM1O165d4eDggH379gEAvvvuO3z00Ue4ffs2TExMGnTM//znP9i0aRMSExMhCEJjhlujgwcPIiEhASEhIbC2tkZsbCwyMjIwatQouLi41LpvUlISoqKi4OzsDDMzMygUCiQmJiI9PR2PPPIIOnToAAAoLCxEeHg4jIyMEBgYCGNjY9y+fRuxsbFo27YtwsLCmuJSGx27ARMRERERUbNSWlqKS5cuoX///uqylStX4oknnmhwogoA48ePx82bN3H48OHGCLNOGRkZiI+PR48ePdCrVy906NABjz/+OCwtLXH69Ok69/f09MSIESPQtWtXdOjQASEhIRg5ciTs7e0RGRmprhcXF4fy8nIMHz4coaGh6NChAwYOHAhfX1/cvHmzxXZ9ZrJKRERERETNxksvvQRTU1MolUq8//77EAQBrq6uiIyMxNChQ7Xqp6SkwMTEBC+++KJG+YEDB2BoaIjZs2ery7p27Qo7Ozts375d79cBAAkJCRAEQd0CCgAGBgbw9/fH7du3UVhYWO9jymQymJubo7y8XF1W9bOZmZlGXTMzMwiCAJmsZaZ9LTNqIiIiIiJqlZ577jlMnz4dAPD999/j119/xYwZMwAAXbp00arv7u6Ol19+GWvXrsXNmzcBAFevXsW4cePw2GOP4ZtvvtGo36VLF5w4caLOOFQqFUpLS3X6qmlkZXZ2NqytrWFkZKRR7uTkpN6ui4qKCpSWliI/Px+RkZG4desW3N3d1dvd3NwAAEePHkVWVhYKCwsRHx+P6OhoBAUFwdDQUKfzNDctc6QtERERERG1SoMHD8bBgwdhbm6OmTNnQiaT4YMPPgAAeHt7V7vPO++8g+XLl+Orr77CZ599hpEjR8LLywsbNmzQalX08fHBr7/+Wmcc6enp2LVrl04xT5gwAZaWllrlxcXFWq2dwN0W0KKiIp2Of+rUKcTExAAABEGAl5cX+vbtq97epk0bdOvWDRcuXFAn7ADQuXNndO/eXadzNEdMVomIiIiIqFmJjIxEUFCQOtHMzs6GgYEBLCwsqq3v7u6OadOmYdmyZTh//jxKSkpw9OhRmJuba9W1tbVFSUlJjYlkFXt7e4wYMUKneE1NTastVygUkMvlWuVVZUqlUqfjh4SEwNvbG8XFxUhISIAoilr7WlpawtXVFd7e3jAxMUFSUhIuXLgAU1NTBAcH63Se5obJKhERERERNSuXLl2q9wy2b731FhYvXozIyEj89ddfGt1k71XVZbeu2YCNjY11Wl6mNgYGBtUmpFVl1SWy1bGxsYGNjQ0AwM/PD7t378a+ffswevRoCIKA69ev49ixY3jmmWfUCb23tzdEUcSZM2fQvn37B5qYSiocs0pERERERM1GXl4ebt26hZCQEHWZvb09FAoFCgoKatzv888/B1DZmmlnZ1djvdzcXJiZmdXYGlpFqVSiuLhYpy+VSlXtMczMzFBcXKxVXlVWXcuvLnx8fJCZmYk7d+4AAKKjo+Hg4KDV8ty2bVsoFApkZWU16DxSY8sqERERERE1G1VLsnTs2FFdFhAQAABITEzUKK+ycOFCLF++HIsXL8bbb7+Nzz//HMuXL6/2+ImJiRqz89bk9u3bDzxm1d7eHqmpqSgvL9eYZCkjI0O9vSEUCgWAu7MAl5SUwNjYWKteVRJd0wRQzR2TVSIiIiIiajYuXboEQDNZ7d27NwDg3LlzWsnq77//jnnz5uGzzz7Da6+9hri4OPz444947733qp2Q6fz583juuefqjKMxxqz6+PggMjISMTEx6NSpE4DKFttr167ByclJ3RKqUChQWFgIExMTje66JSUlWsdWqVSIi4uDXC6Hra0tAMDa2hrJycnIy8tTdxcGgPj4eAiCUGtLc3MmiC01zSYiIiIiolZn2rRp+OOPP5CcnKxRHhISgpCQEKxfv15dFhERgf79+2PMmDFYs2YNACA1NRU+Pj6YNGmSVutqREQEunXrhgMHDmDIkCH6vxhUrveamJiIkJAQWFtbIzY2FhkZGRg5ciRcXV3VMe/atQtdunRBt27d1Pvu378f5eXlcHV1hbm5OYqLi3H9+nXk5eWhV69e6sQ9LS0Nu3btgomJCYKCgmBsbIykpCTcunULAQEB6N+/f5Nca2PjmFUiIiIiImo2IiMjq+3q++KLL2Lnzp0oKSkBACQnJ2PUqFHo3Lkzli1bpq7n5uaGF198EWvWrEFiYqLGMX777Td4enpi8ODB+r2IewwcOBAhISGIi4vDyZMnoVKpMHz4cHWiWhsfHx8IgoDo6Gj89ddfuHz5MszNzfHoo49qPEeurq548skn4eDggCtXruDvv/9Gfn4+unfvjn79+unz8vSKLatERERERNTs3blzBz4+PliwYAFeeumleu9fVlYGLy8vzJs3D//+97/1ECE1NrasEhERERFRs2dtbY25c+di4cKFNc6+W5uVK1fC0NAQM2bM0EN0pA9sWSUiIiIiIqJmhy2rRERERERE1OwwWSUiIiIiIqJmh8kqERERERERNTtMVomIiIiIiKjZYbJKREREREREzQ6TVSIiIiIiapHy8/MxcOBA5OfnSx0K6QGTVSIiIiIiapHy8/Nx9OhRJqutFJNVIiIiIiIianaYrBIREREREVGzw2SViIiIiIiImh0mq0RERERE1CJZWVmhd+/esLKykjoU0gNBFEVR6iCIiIiIiIiI7sWWVSIiIiIiarEuX74sdQikJ0xWiYiIiIioxVIqlVKHQHrCZJWIiIiIiFosOzs7qUMgPWGySkRERERELZatra3UIZCeMFklIiIiIqIWKz4+XuoQSE+YrBIREREREVGzw6VriIiIiIio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OgTSE/51EREREVGtXFxcqi13tRAwzv/hTle7OD/c198cWFlZSR0C6QnHrBIRERFRg5UqRJQopI5COvHRl9CtS6jUYTzUiouLYWZmJnUYpAcGdVchIiIiIqqeiYEAk4f4E6UB+ylKLjY2FqGhoVKHQXrAPy8iIiIiogby8vKSOgSiVovJKhERERFRAxUVFUkdwkPP09NT6hBIT5isEhERERE1UGZmptQhPPTKysqkDoH0hMkqERERERG1WLdv35Y6BNITJqtERERERA3UqVMnqUMgarW4dA0RERERUQNFR0cjMDCw2m1KlYhyZRMHpAMTA0AQWs/6sEqlEnK5XOowSA8e4onGiYiIiIgeTHl5eY3b3j+uwn/PNL92ofn9ZHivV+tJVuPi4hAQECB1GKQH7AZMRERERNRAVlZW1ZaXVIhYcrH5JaoAcCyxROoQGlVpaanUIZCeMFklIiIiImogFxeXasvXXxVxp+ZGV0mVtrLZcy0tLaUOgfSEySoRERERUQPFxsZqlYmiiG/OqtBcO9oaGxtLHUKjcnd3lzoE0hMmq0REREREjejoLRExOUDz7AQMyGWtKwW4evWq1CGQnrSuVyoRERERURPy9PTUKvvuvAoGzbVZFUBxSesas0qtF2cDJiIiIiJqoLL7xn/euCNix/Xm26raGnl4eEgdAukJW1aJiIiIiBro9u3bGo//74IKsmbcqgoAFhYWUofQqJTKZriYLTUKJqtERERERI2gqFzET5EilM28WfX+1uCWLi0tTeoQSE+YrBIRERERNVBISIj657UxIgqb6XI196qoqJA6BCKdMFklIiIiImqguLg4AJXL1Xx7TiVxNLqRtbLZgIOCgqQOgfSkdb1SiYiIiIiaUGlpKQDgYJKI2NyWMbGSpaWl1CE0qsTERKlDID3hbMBERERE1OzklYqYe0yFbXEiiiuAHq4CvhkoQxdn3WYviskWMfuwCsdTRBjJgcd9BHw7UAZHs7v7pxaKmHtUhbPpIlILAbkM8LMFXguVYXKQAEGo+1xVid93ESrIBTT78aoAcOfOHQB2UofRaIqLi6UOgfSELatE9EAEQUD//v11qrt06VIIgoCffvpJz1EREVFLphJFPL5VifUxImZ2lmHBABkyikUM3KREXG7d2WBygYj+G5W4nifii0dkeKubDLsTRAz7TYnye7LJrJLKumP9BHw9UIb5fWVwNRcwZa8K7x3XrUuvu7s74vNE7EloGYlqa2Rubi51CKQnbFmlVuvnn3/Wue6ECRNaXZeY+928eRMff/wxDh8+jNu3b0OhUMDGxgYhISGYPHkyJk+e3OrGsBARUcsUfk3EyVTgt1EyjPWv/N803l+A3wolPjqhwvqR8lr3/+KUCkUVQMTzcnhaVbaO9nAFhv2mwqooEa90qizr6CjgyLOaH4dndgFGbVXih/MiPusrQl7HOjRXr17FqpwQyFpIqyoAGBkZSR1Co2rbtq3UIZCeMFmlVmvQoEEaj9PS0nD16lUEBATA1dVVY5uJiUlThtbkNm/ejClTpqCiogL9+vXDhAkTYGFhgRs3buDPP//E1KlTkZOTgzlz5ug1jpdffhmTJk1q9c83ERE9mPBYEc5mwNN+dxNFRzMB4/0FrI0WUaYQYWxQcxK5JU7ESB9BnagCwNC2MvjZqrD5mohXOtV+fi9roDgBKFcCpnXcxy1SyLD8cvNfruZeBgatKwWIjo5GaGio1GGQHrSuVyrRPXx9fTUeq1QqXL16Fc7Ozlrb7ldeXt5q7jpeunQJzz//PMzMzLB9+3YMGzZMY7soiliyZEmTJJAGBgatbiFyIiJqfBcyRHRxFiC7b8xoD1cBP0dWTmQU4lj9vikFIjKKgW4u2slsD1cBexK0s8qSChFFFUBhBXD0loiVUSJ6uwGmhnWPWT1e4oXiFrYSTOUYT944puaPff7oobd+/Xrs3LkTWVlZ2LNnD1auXIktW7YAAM6dO4eff/4ZBQUFNe53v+TkZOzevRurVq3CihUrEB4ejujoaL1fR03eeustlJeX46OPPtJKVIHKMaevvvoqXnzxRXXZ559/jq5du8LOzg4GBgawsbHBoEGDcPr06RrPs337dnTo0AHGxsawtrbGU089hdTUVI061Y1ZrSpbsmQJ3nvvPbi4uMDQ0BAuLi744IMPoFK1jGUAiIio8aQVAa7VDEOsKkstrLkZM61Is+79++eUAmUKzf2/Py/C8UclvJcpMWWvCr1cBWyso6sxUDm2dskV4zrrkX65ublJHQLpCVtWiQAUFhZi165d8PHxgbe3d4MXy46JicFff/0FZ2dndO7cGQYGBkhJScHx48eRn5+PXr161XmMsrIyiKJufYkMDQ0hl9f8z7S0tBRHjx6Fra2tRjJal8WLF8Pf3x/PPfccHB0dERMTg23btiEsLAwnTpzQWs8sMTERzzzzDEaMGIGxY8fixIkT+P333xEbG4vTp0/r1Jq6aNEilJaWYvz48TA3N8fatWsxf/58+Pj4YOrUqTrHTkREzYtKFFGu1K2usbzyJmqJovLn+5n888m1RFHzMaq2Vbu//G4d43s+BU8IENDNRYbMYmBXgojbRbWfo8r+GyJuFtad1DY3rW1CIl1mbaaWiS2rRAAKCgrQq1cv9O/fHx06dEDHjh3rfYzi4mKcPHkS7dq1w5NPPolOnTohKCgIjz76KIKDg3H58mXk5+fXeZwtW7ZgzZo1On1dv3691mPFxcWhoqICbdu2hZWVlc7XEhMTgyNHjuB///sfPvzwQ2zYsAFbtmzBnTt38M0332jVT05Oxscff4ytW7fis88+w6FDhzBlyhRER0fj66+/1umcpaWluHDhAn744Qd8+eWX+Pvvv2FgYFCvibIAICcnB2VlZerHhYWFGi3j5eXlyM7O1tgnLS2t1sfp6ekaNxB4Dp6D5+A5eA7dz3EwsQKm3yl1+rqWU7mPiVxE6T0JbtU5SqsSSEVpjddh+k8Sejs7T+s6qo5paqB5HW2tBXSxzMPTPhVY97gcPjbAkM0KZOTW/lydSMhHy1hZVVPV2rBVWuLr6t5zpKSktIrreNjOoQtB1LUJh6iFu3btGo4ePYoBAwbA399fXb5+/XpUVFTg+eef15oN99y5czh//ny1swWvX78elpaWGDVqFAAgKioKJ0+exOOPPw57e3uNulVdjB955BF06NCh1jjT09OhUOhwOxeAnZ0dzMzMatx+4sQJ9OvXD3369MGJEyd0Oua9VCoVcnJy1P/UAgICEBAQgHPnzqnrCIIAZ2dnJCQkaMSSlpYGNzc39OjRQ919eOnSpfjXv/6FpUuXYvr06Rplc+bM0UqE27VrB4VCgevXr8PQ0LDe8RMRkfTSi0TsTdTt4+ZTvgKsjQX4LlfA11bAnjGarZYrLqvw8j4VIl+QI8Sx+ta0lAIRHj8p8VV/Geb20Py//vweJfYkiMieWXvnwv03VAgLV2HvGBnCvGtu28ksFuG+VIEKVctq2evvkI+jU1rPOqsXL17kBEutFLsBEwGwsrJ64GVb8vLyAAC7d++usU5JSUmdx3FxcXmgOO5V1Zpa38Wyd+7ciffffx8xMTFaXaILCwu16ru7u2slza6urrCwsMDt27dRUVFRZ7Lp4+OjVWZjY4ObN2+itLSUySoRUQvlYi5gSnD9krlQJwF/JYtQiaLGJEun00SYGQB+tjXv624pwNEUOJeunSCfSRMR6lR3LFVdgO+U1V7P0UzAKPdC7Ei2hKIFNf+0tm6zdTUEUMvFZJUINU/hXtub+f2dEqoeDxw4sMbWTl264paUlOg8ZtXIyKjW6ed9fX1haGiIpKQk5Ofn63T+v//+G6NHj4aTkxNef/11+Pv7w8LCAoIgYPr06TrHVl+tbRp9IiJquLF+AsJjRWyNFTHWv/J/cVaxiN+uiRjVTtBYtiY+r/L/Ujubu2Vj/ASsviLiVr6INv8sX3PwpgqxucDsrnfrZRaLcDTT/l+/4rIIAUAX57qTunGut7H1Vstaq70+Q4Naglu3bqF9+/ZSh0F6wE+HRLUwNq6c4a+srEyjG7BCoUBxcbHGm721tTWAyjVbPTw8GnzObdu2Vdt6WZ37uzTfz8TEBP3798fBgwexatUqzJo1q85jLl++HCqVCmvXrsWQIUPU5UVFRZg0aVK1+6SkpKC4uFirG3BhYSECAwPZKkpERPUy1k9AL1dg6l4VorMBB1Pgx4sqKEXgk76aPaGGbK4ciHrjlbsfa9/tKcNvsUoM2qzEv7vIUFgBLDyrQogDMPWeVt7PT6lwIkXEcO/KNVlzSoAtcSqcTQde7yygvW3dyWpXVzn6uAGn09Bi1lqtnEOjlubpFkbXz03U8jBZJapFVQKanJwMBwcHdfnly5e1Whh9fHxw9uxZREREwM3NTaulsLy8HHK5vNbZewFg8ODB9RqzWpevv/4aPXv2xCeffIKgoCCNBLTK0qVLYWRkhBdffFEd3/1LxsybN6/GZWRu376NH374AfPmzVOXvfvuuwCAsLAwna6FiIioilxWOV717aMq/HBBhZIKoLsLsOoxOfzt6k4g21gJOPqMHHOOqDDvmApGcuBxHwHfDJRptMo+7iMgPg/4JUpEZrEIEwOgoyOwcrgMLwTp1lW2bdu2mCPKMHZHy1lqrbVNWdMUa8WTNJisEtXC3d0d1tbWiIiIULeupqenIyMjQ+uN0cLCAv369cOxY8ewefNm+Pr6wtLSEiUlJcjJycGNGzcwfvx4rYma7teYY1YBIDQ0FL/++iumTJmC4cOHo3///ujRowcsLS1x8+ZNHDhwAAkJCVi0aBEAYMKECVi+fDkmT56MZ599Fubm5jh69CiuXr1aY+weHh74+OOPcfbsWQQGBuLEiRM4fPgwAgICMHfu3Ea9HiIiejjYmghYHibH8jrued7bonqvIAcB+8bWfoN4mJcMw7waGOA/oqOj8WTHTnA1v7vGa3NnZGQkdQiNil2AWy8uXUNUC5lMhrCwMLi6uiIqKgpnzpyBSqXCqFGjqh1j6e/vj1GjRsHBwQExMTE4fvw4rly5guLiYnTv3h2mpqYSXAUwfvx4xMTE4LnnnsP169exaNEifPTRR9i6dSvatm2LNWvWqLsIDxo0CKtXr4a5uTl+/PFHfP/99xAEAXv27KlxLK63tzc2bdqEK1eu4KuvvkJERASeeOIJ/PnnnzqtsUpERNSSGcgE/LuLrMV8sDZsZfNEREVFSR0C6QmXriEiIiIiaqCMjAw4OTkhu0SE+1IlypR17yM1Ll1DLUVLuQFERERERNTsVK0cYG8q4PlAAQata1WYFqGxh1BR88FklYiIiIiogVJSUtQ/z+oiaxHrrdY0rKel4qoDrReTVSIiIiKiRhDiKKC/ByBv5q2ruq460FLcunVL6hBIT5isEhERERE1UIcOHTQez+kma/brrZaXl0sdApFOmKwSERERETXQ/a16I30EeDTzifCbecNvvfn7+0sdAukJk1UiIiIiogYqLCzUeCyXCXijq6xZJ4RW1tZSh9Co0tLSpA6B9ITJKhERERFRA5mYmGiVvRQiwFguQTA6KsjPlzqERpXfyq6H7mKySkRERETUQO3bt9cqszERMCW4+S5joxKb+aDaejI2NpY6BNITJqtERERERA0UFRVVbfnrnZvvMjatbakXjlltvQykDoCIiIiIqLUJdBAw2BM4cqt5LWWjUAHmJkZSh9GoIiMjERoaKnUYpAdMVomIiIiIGsjFxaXGbQv6y7Erofk1r3YXrgEIkjoMojoxWSUiIiIiaqDaktWuLgK6ujSjZtV/XLxYIXUIjcrJyUnqEEhPOGaViIiIiOgh4uXlJXUIjcrU1FTqEEhPmKwSERERET1EioqKpA6hUd28eVPqEEhPmKwSERERET1EMjMzpQ6BSCdMVomIiIiIqMXy9fWVOgTSE0EUW9mqwEREREREVCNRFCEIzW/ip4a6ceNGqxuHS5XYskpERERE9BCJiYmROoRGlZeXJ3UIpCdcuoaIiIiI6CFSXl5e47Zfr6hwNLlpOl6aGQDz+8lgZfxgrbyGhoaNFBE1N0xWiYiIiIgeIlZWVtWWlylE/PuwCnfKmqb7pUIE/GyVmNnlwVKSoKCgRoqImht2AyYiIiIieoi4uLhUW77pmojcUkAlViaS+v4SAKSmpD7w9Vy6dOmBj0HNE5NVIiIiIqKHSGxsrFaZKIr45pwKLXHaJc4X23qxGzARERER0UPuZCoQKcHyqxaWlg98DAcHh0aIhJojtqwSERERET1EPD09tcq+i1DBQIJmVUPDB287s2yEhJeaJyarREREREQPkbKyMo3Ht/JFbI0ToZCgN21uTu4DHyMxMbERIqHmiMkqEREREdFD5Pbt2xqPl1xqmWNVqfVjskpERERE9JAqqRDx40URSonmKHJydnrgY7Rr164RIqHmiMkqEREREdFDJCQkRP3zhqsi7pTVUlnP8vMLHvgYubkP3pWYmicmq0RERERED5G4uDgAzWO5mtKSkgc+Rk5OTiNEQs0Rl64hIiIiInqIlJaWAgCOJQPR2dLGIpfLm8UxqHliskpERERE1AjySkXMPabCtjgRxRVAD1cB3wyUoYuzbm2XMdkiZh9W4XiKCCM58LiPgG8HyuBodnf/q9kifolSYf8NEfF5gIUR0MVJwCd9Zejmott5qpZ6+S5CCQMBkswCXMXV1fWBj3Fvt2ZqXdgNuAVwcXGBv79/k5+3a9eucHBwQEHBg48lqMvSpUshCAJ++uknvZ9LSlL9LomIiEi/VKKIx7cqsT5GxMzOMiwYIENGsYiBm5SIy607G0wuENF/oxLX80R88YgMb3WTYXeCiGG/KVF+z+xHyy+rsCxSRDeXykR4TlcZruWK6LVOiQM3VTrF6u7ujht3RGy/Lm2iCgDJyckPfIzLly83QiTUHOncsvrzzz/rfNAJEya02sV533zzTXz77bc1bg8ICEBMTEwTRkT3+uOPPzB//nzExsYiLy8PpqamcHR0RKdOnfD222+jd+/eeo/hww8/hCAI+OSTT/R+LimdOXMGy5cvx0svvYSePXtKHQ4REZGkwq+JOJkK/DZKhrH+le1B4/0F+K1Q4qMTKqwfWXtX1S9OqVBUAUQ8L4enVWULaQ9XYNhvKqyKEvFKp8qyCQEyfNwHsDC624r6YoiADr8o8fFJFYa2rbst6urVq1h/JwQyAZLNAtyYlEql1CGQnuicrA4aNEjjcVpaGq5evYqAgACt5nsTE5PGia4ZmzJlCoKCgrTKnZwefPptapgFCxbgP//5DxwdHTFq1Ch4eXkhOzsb165dw4EDB9C5c+cmSVZ//vlnWFtbt/pk9fz581i2bBm6du3KZJWIiB564bEinM2Ap/3uJpGOZgLG+wtYGy2iTCHC2KDmbrpb4kSM9BHUiSoADG0rg5+tCpuviXilU2VZ12q6+tqbCnjEQ8CRW7plniVKAUsvSbdczb3MLcwf+Bh2dnaNEAk1Rzonq76+vhqPVSoVrl69CmdnZ61t9ysvL4eRkVHDImymhg0bhokTJ0odRqMTRRG5ubkt7o9eoVDgs88+g52dHc6cOQMvLy+N7eXl5cjIyJAmOCIiImr1LmSI6OIsQCZoJpM9XAX8HCkiNhcIcax+35QCERnFqHbMaQ9XAXsS6s4q04tEOJjqFuvJUm8UlutWV9+MjY0f+Bi2traNEAk1R40+ZnX9+vXYuXMnsrKysGfPHqxcuRJbtmwBAJw7dw4///xztWMgq/a7X3JyMnbv3o1Vq1ZhxYoVCA8PR3R0dGOHrRdV4zB//PFHvPHGG3B0dISRkRHat2+PTZs2AQD27t2L4OBgGBsbw8bGBlOnTkV5efXvHidPnkTnzp1hYmICc3NzDBkyRKvLcUFBAaZPnw5/f39YWlrC0NAQzs7OmDBhAm7fvq1R98iRIxAEAfPmzcPnn3+ONm3awMjICK+99lqN15SZmYng4GCYmppqjC/dvn07unXrBnNzcxgaGsLT0xNz585FRUWF1jG+/vpreHh4qGObNWtWtfUAICIiAhcuXIBKVfsYjKysLBQWFsLHx0crUQUAIyMjeHh4aJQVFRXhlVdegYuLCwwNDWFtbY3Bgwfj7NmzGvVqG097/7heQRBw+/ZtxMbGQhAE9dfBgwc19ouMjETv3r1hamoKMzMz9OnTB1FRURp1UlJSMGnSJPj4+KifV3d3d8yYMQNFRUXVxvigr7WqMbV1vdbefPNN/Otf/wIAzJgxQ32d/fv3V9fJyMjAM888AwcHBxgYGMDe3h5PPPEErl+/Xm3sS5YswXvvvaf+fbi4uOCDDz6o83dPRETUHKQVAa7VNBJWlaUW1pxwphVp1r1//5xSoKyWwaV/JYv4OxV4xr/uCZZEUcSPUc2nESkn+8GXnYmPj2+ESKg50stswIWFhdi1axd8fHzg7e1dYyJSl5iYGPz1119wdnZG586dYWBggJSUFBw/fhz5+fno1atXnccoKyuDKOrWx8HQ0FDnqa/v3LlT7YBwS0tLWFtba5R9++23UCgUmDRpEkRRxMqVKzFlyhRkZ2fjP//5D5544gk8/fTT2LlzJ1atWgU3Nzd8/vnnGsfIzc3Fo48+ir59++Lxxx/H5cuXsXPnTgwdOhSnT59WJ2IpKSnYtGkTBgwYgCeffBLGxsY4efIkNm3ahCtXruDMmTNa3bQ3btyI/Px8jBkzBh4eHnBxcan2muPi4jBo0CAUFhZi48aNePLJJwEA33zzDd5++220b98eL7/8MqytrXHkyBEsXLgQcXFx2LJlC2Syyvsin332GT788EN4enritddeg0KhwPr162FlZVXtObt16wZ7e3skJibWOg7a2dkZpqamuHbtGg4ePIghQ4bUWBcAKioq0KdPH3XS+NJLLyExMRHh4eEYOnQo9u/f36Curd9//z0++OADmJmZ4c0331SXBwYGqn/Oy8vDgAED0KdPH7z99tuIjo7G1q1b8eyzz+L8+fPqXggXL17Evn37MGjQILRv3x4qlQqHDh3CTz/9hPj4eOzdu1fr9dpUr7Xx48cjKSkJ4eHhePbZZ9G1a1cAgI+PD4DKv48uXbogNTUVw4YNQ/fu3REVFYUdO3bg3LlzOHnypNZNhUWLFqG0tBTjx4+Hubk51q5di/nz58PHxwdTp06t9++CiIiooVSiiHIdh0EayytvVpcoKn++n8k/n7ZLFDUfo2pbtfvL79YxruaTe0aRiIm7lfC2Bub2qLsd6lCSiIQCLghCLYNeXqkFBQXo378/AgICGnyM4uJinDx5Eu3atdNIPIKCgnDy5ElcvnwZgYGBNSY5VbZs2YLCwkKdzjlgwACdZ2p99dVX8eqrr2qVv/TSS1i+fLlGWUVFBc6fPw97e3sAQN++fTF+/Hi8/vrrWLt2LSZMmAAAeP/99+Hq6oqNGzfi448/hqGhofoYmZmZmD59OpYsWQLhn+4lVYnf+++/j1WrVgGoTBZSU1NhZmamEcMbb7yB77//HuHh4Zg0aZLGtszMTJw4cQKhoaE1Xu+pU6cwYsQIGBsb448//lCP/UxLS8M777yD3r174/DhwxrdvadOnYrVq1fj0KFDGDp0KPLy8vD555/D1dUVf//9N9zc3AAAb7/99gO9VoDKfxJz587FJ598gkcffRTe3t7o1KkTevbsiSeeeELr+D/++CMiIyMxevRohIeHq5O+MWPGYOzYsXjzzTdx7NgxdZKtq1mzZuGLL76AlZUV3nrrrWrrZGRk4JNPPsGHH36oLnv++eexdu1a/Pnnn3j88ccBVI4TT01N1XgdAMATTzyBXbt24cyZM1pjcJvqtdazZ08MGTIE4eHhGDhwIKZPn64Rx/vvv4+UlBS8/vrr+P7779XHqRpX/J///Efd4lultLQUFy5cUMf+2muvwdvbGz///DOTVSIialLHbokYtFm3nj0xU+UIsAdMDYCyahLc0n8SUdNaPnVXbat2f2XN+xeVixi5TYmCcuD4s3KNSZdq8kuUCBlEqKDbMjf65uhYQ9/oevD29m6ESKg50svSNcbGxvDz83ugYyQkJECpVCIgIAClpaUaX56enhBFESkpKXUeZ/DgwRgxYoROX23atNE5vjfeeAPr16/X+nr99de16o4dO1b9ARwAhg4dCqDyD2v8+PHqciMjI3Ts2BG3b9/W6iptYmKC9957T/2hHwDmzp0LMzMzHD16VN16bWRkpE5UKyoqcPv2bSQnJ6sToL///lsrvkGDBtWaqO7evRuDBw+GnZ0djh49qpEgrV69GhUVFZg0aRIyMjKQnJys/ho7dixEUcTu3bsBALt27UJZWRnGjBmjTlQBoG3btupW2vuJooisrCydZpf++OOPsW7dOnTv3h3JycnYunUr/vOf/yAoKAjDhg1Denq6uu5vv/0GQRDw8ccfa7ROjhkzBr6+voiIiEBmZmad52wIGxsb/Pvf/9YoGzFiBABodLU1MzNTJ5GlpaVIS0tDcnIyRo0aBVEUcfz4ca1jN+VrrTY7d+6EpaWl1nHmzJkDa2trHD9+HCUlJRr7jBs3TiN2Dw8PeHp6IjU1tV69M3JyclBWVqZ+XFhYqHGN5eXlyM7WXAE9LS2t1sfp6ekaPTR4Dp6D5+A5eI7WfY4AewFLBlZgycByrBwuw8rhMiwbqsL/+hWrH68cLsOiXvlwtajc39UcuJFTqnWOqu6/bhZCjddR1f33Vr5C6zoSMktgZwL15ExV11GuFPH0DhUiM0T8/qQMwY6CTs9VX1clmtMAm6qhTQ/yO68qb+6vK55D87Eu9NKyamVlVe8Wqfvl5eUBgDrRqc79H3arU1OX1gfVvXt3dStVXaq6RlapGgTu4uKi1Y3TxsYGRUVFKC3VfLNzdHSEs7OzRpmxsTHc3d2RlpaGgoIC9aRIX3zxBX7++WckJSVpdYGuel5ri+9e+fn5mDRpEtzc3PDXX39pzfxcta5VTS3NANRJ39WrVwGg2lbUjh07arW0NcTEiRMxceJEKJVKnD17Fjt27MCKFStw4MABPP/88+qus7du3YKNjU21d+ICAgIQFxeHmzdvaj3njcHBwUEr+a56XrOystRlCoUCc+fOxcaNG7XeMIDKN437NfVrrSZpaWnw9PTUultqYGAALy8vREdHo6CgAKamd2eCqO51aGNjg5s3b6K0tFSrhbkm98dmYWGh8djIyEgjKQa0FyS///H97yM8B8/Bc/AcPEfrPoeLuYAZ3e6frcjony+NvdQ/hToJ+CvZGCpRVE+y5OLigjORSpgZAH62gLFB9ddhCcDRFLiYKYd9b83ruHzHBKFOd2/8urq6QiWKmLxHhYM3RWweJcdAz7ufu+t6rqaEGGLusQoUKZpHy2pxcTGAB/udZ2VlwcPDo9m/rngOzce60EuyamBQ/WEFoeY/ivs/iFc9HjhwoFaX1ip1dQEGKhNaXcesGhkZ1Rj7g6jpmA+a0Ffn008/xUcffYSOHTviww8/RNu2bWFiYoL09HTMmTOn2slqanp+AcDc3ByBgYH4+++/8c0332DhwoXV/h4//PDDGrvytm/fvuEX1EByuRy9evVCr169MHv2bHh5eeHIkSNIT0+Hu7t7o52nIet6yWQynX73L7/8MlavXo2+ffti5syZcHNzg7GxMS5fvowvv/yy2td1U77WGps+/vaIiIiaylg/AeGxIrbGihj7z0RHWcUifrsmYlQ7QWPZmvi8yv/h7Wzulo3xE7D6iohb+SLa/LN8zcGbKsTmArO7an72ev2gCpuuifhpmAxP+9Xvf7yZoYBx7tn4NcmxWSxdIzTCZ5Tacgxq2Zr002HV1NRlZWUaLUsKhQLFxcUayWfVJEUmJiZas7jWx7Zt2/QyZrWpZWZm4vbt2xpdlcvKypCSkgInJyf187l27VrY29vj6NGjsLGxUdfdtWtXg84rl8uxc+dOjBs3Dt9++y0qKiqwaNEidfJTlaDa2NjU2dJc9dxWtbDeKzIyskHx6cLR0RFt2rTBtWvXkJycDHd3d3h6euLEiRO4ceMGOnbsqFH/2rVrMDExQdu2bQHcXTv3/q4NAKqdZKux3jC3bNkCPz8/HDlyRCORa0gXivrQ9bVWGzc3N9y+fRuZmZkarbQKhQI3btyAo6OjTschIiJqKcb6CejlCkzdq0J0NuBgCvx4UQWlCHzSVzMhG7K58mb3jVfu/n9/t6cMv8UqMWizEv/uIkNhBbDwrAohDsDU4LufLb6LUOHHiyJ6uwFmhsDaaM2GiKfaCzCvY+zqC36lWH3zQa+4cbjfMzSsoTp16tQIkVBz1KTNLVUJ6P0f8C9fvqzVSuTj4wO5XI6IiAgoFNrTp5WXl+vUqqWvMatNrbS0FJ9//rnG87Rw4UIUFxdjwIAB6i6ScrkcgiBo1FMoFPjoo48afG5DQ0Ps27cPw4YNww8//ICZM2eqW2inTJkCQ0ND/O9//9NaGgeonFn2zp07AICRI0fCyMgIW7ZsQWpqqrrOzZs3sX379mrPrevSNcXFxfjtt9+qrRcTE4OEhARYWFigXbt2AKAeT/vJJ59o7LNt2zbExsaia9eu6i6sQUFBAICjR49qPK+//PJLtQmsmZlZtd1r60smk0EURY1zFhYWYtGiRQ903Lro+lqr+nuu7jkYOXIkCgoKtFqAv/vuO9y5cwf9+vXT6AJMRETU0sllAvaMkeOZAAE/XFDh7aMqOJgCh8bL4W9X943sNlYCjj4jRzsbAfOOqbDgjAojvAX8OU6u0Sp7MaPy/+rfqcDze1RaX5l1j5JDnw7uGN0ekDeDBkld5qCpy5UrVxohEmqOmrRl1d3dHdbW1oiIiFC3rqanpyMjI0NrORULCwv069cPx44dw+bNm+Hr6wtLS0uUlJQgJycHN27cwPjx4+tsndHXmNU///xTI+GqYmpqildeeUXn8XW6cnR0xNq1a5GYmIgePXogMjISO3fuhKurKz799FN1vaeffhpffPEFBgwYgCeeeALFxcU1JoL1YWBggD179uDJJ5/EkiVLoFAosHTpUrRp0wZff/013njjDQQFBWHkyJHw8fFBdnY2oqOjcezYMezZswdDhgyBra0t3nnnHXzyySfo3bs3nn76aSgUCmzYsAGurq5ITEzUOq+uS9cUFxdj/Pjx8PDwQL9+/eDn5weZTIZr165h586dqKiowBtvvAEHBwcAlWNsf/nlF2zduhUDBgxA//79cfPmTWzevBlWVlb4+uuv1a3H/v7+6NmzJ/7880+MHDlSvQzLgQMH4OrqqrVWadeuXREeHo6XX34ZnTt3hlwuxzPPPFPvfvojRozAxo0bMXDgQAwbNgxZWVkIDw/XGiPQ2HR9rfXt2xeCIODnn39GRUUFrK2t4eXlhdGjR2P+/PnYtm0bfvjhB1y9elX9nG3fvh0uLi746quv9HoNREREUrA1EbA8TI7lYbXXu7dF9V5BDgL2ja19GcVVj8mx6rGGRlgpOjoab3TthG3X6z+cqbHpOlyvNg1dJpOavyZNVmUyGcLCwnDy5ElERUVBLpfDw8MDo0aNqjah8vf3h7W1NSIjIxETE4Py8nKYmJjA2toa3bt3l7RlpmqpmPvZ2Nhg0qRJWmutPihbW1v8/vvveO2117BgwQIYGBhgwIABWLx4MTw9PdX1Pv30U1RUVODXX3/FV199BWtrawwZMgRvv/02unfv/kAxyOVy7NixA2PGjMGyZctQUVGB5cuXY9asWejQoQM+++wzbNu2DUVFRbC0tIS7uztmzJih0TXj448/hpmZGb7//nssXrwYdnZ2mDhxInx9fTFr1qwGx2ZjY4Ovv/4af/zxB44ePYrff/8dFRUVsLS0RHBw8P+3d+dxUVX9H8A/dxj2fZUdJNncwH3LXXNLc7f0abHSrGxxSXu0NMu0tNWWn6ZppU/mrrll4ZL7EoqKoAhuICAIIrIOM3N/fxCj4wwwwCyMfN6v17xkzj33nHMPV+A759xzMGnSJDz33HOq/JaWljh69Cjefvtt/P777zh27Bjs7Ozw+OOP49NPP9Xoq82bN+M///kP9u3bh7179yIyMhKbNm3CjBkzcP26+jyab775BtnZ2di6dSt+/fVXiKKIFi1a1DhY/fHHH2Fra4vt27fj5MmTcHd3x1NPPYWhQ4eqVg82BF3vtcDAQHz99ddYtGgRPvroIygUCnTt2hVDhw5VfSj1xhtvYO/evYiJiYGTkxMGDRqEL774QmOPVSIiIjKurv5AM3cgIQcw5aOrdvaVr52iqwcffaNHiyDq4+MMInokeHt7w9nZGZcuXTJ1U4iIiMhAsrKy4OXlhVXnlXhxj+k2shEALO4iw7ROdQtYCwsLYW9vr59GUb1S/5cIJSIiIiIivalYDPKZSAEu1qZtS84D2/bV1uXLl/XQEqqPGKwSERERETUgFYsa2UgFvBYt1IuFloi0YbBKRERERNRAvRotMekzq+4e7nUuo2K7QXr0MFglIpXMzEw+r0pERPSIi4yMVH3t7yhgRKgAqYlGV0uK67bVHwAUF+uwXw+ZJQarREREREQNSGpqqtr7t9tIIDfR8GphYWGdy8jKytJDS6g+YrBKRERERNSAFBQUqL3v5AtEeQISU4yu8nlZqgKDVSIiIiKiBsTGxkbtvSAImNZWAqUJRlf9/fzrXEbLli310BKqjxisEhERERE1IE2aNNFIGx0uwM1GS2YDS89Ir3MZXG/j0cVglYiIiIioAYmPj9dIs5YKmNxKMPpUYKVCWecySktL9dASqo8YrBIRERERESZFSYwaHIgAbO1s61yOk5NT3RtD9ZLU1A0gIiIiIiLj8fb21pru4yDg+WYCNiYZ5+FVW0ugz2N2dS7Hx8dHD62h+kgQRdGU+wATEREREVEDFRcXh+joaJOXQfUTpwETERERERFRvcNglYiIiIiITCI4OLjOZQQEBNS9IVQvMVglIiIiIiKTKCwsrHMZZWVlemgJ1UcMVomIiIiIyCSys7PrXEZmZqYeWkL1EYNVIiIiIiIiqne4GjAREREREZmEKIoQBKFOZcjlckil3JHzUcTvKhERERERmURiYiKaNm2q9VjyHRFF8urLuHr1Bho3blxlHldrIMCpbkExGR+DVSIiIiIiMgmZTKY1PadYRPOfFChV6FJKIHCw6ozWFkDCeAuEuDBgNSd8ZpWIiIiIiEzCyclJa/qKcyLKlPqrp1QBJKTm6K9AMgoGq0REREREZBLe3t4aaXKliK9PK6HU88o6OTkMVs0Ng1UiIiIiIjKJpKQkjbRtySIy6r79qgZXV1f9F0oGxWCViIiIiIjqjS/+UUJigEdLra2t9V8oGRSDVSIiIiIiMonAwEC192duiTiaDr1PAQaAzMxM/RdKBsVglYiIiIiITKK0tFTt/ZLTSki5YC/9i8EqERERERGZxK1bt1RfZxeJ+F+iCLkBRlUBzVFcqv8YrBIRERERkcn9cE6EwkCBKgDk5uYarnAyCAarRERERERkEi1atAAAlClELDHAdjUPKigoMFzhZBAMVomIiIiIyCQuX74MANh8WURWkWHrkkqlhq2A9I7BKhERERERmURJSQkAw21X86CQkBDDVkB6x48XiIiIiIgeMXklImYcVGLLZRFFZUB7HwGf95CgdSPdIsLEHBFT9itx+KYIKwtgUIiAL3pI4Gl3//yLOSJWxivx5zURKXmAgxXQ2kvAvC4StPXWrR5HR0f8kynipBF2lUlKSkIn33DDV0R6w5FVIiIiIqJHiFIUMWizAr8mipjcSoJF3SXIKhLRY50Cl+9U/1Bo2j0R3X5TIDlPxIKuEkxvK8HOKyL6blBA9sAKSCvOK7H8nIi23uWB8NQ2Ely6I6Lj/xSIua7Uqa1+fn74Opbb1ZB2HFklMrC8vDycPn0at2/fRmFhIZRKJRwcHBAYGIioqCjY2dlpPefEiRPIyMiAUqmEh4cH2rRpAz8/PxNcgbrdu3dj4MCBEAQBv/32G0aPHm3qJhEREdEDNl4ScTQd2DBYgpHh5WNTo8MFhP2owNwjSvz6pEWV5y84rkRhGRD7rAUCncqjyPY+QN8NSvwUL2JiVHnaMxESfNAZcLC6H2m+2EJA5EoFPjiqRJ+g6sfFDp9Nxm8XIwy2Xc2DXFxcDF8J6RVHVokMrLCwEEVFRQgODkb79u3RuXNn+Pv7IzExEZs3b0ZxcbFa/vz8fGzbtg1ZWVmIiopChw4dUFZWhl27diEtLc1EV3Hf//3f/8HV1RUWFhb49ddfUVZWZuomERER0QM2JoloZAcMD7sfRHraCRgdLmBbsojSaiLDTZdFPBkiqAJVAOgTJEGYK7D+0v1z23gLaoEqALjbCujqLyAxR7e2brrpCt3GYOvO1tbWSDWRvnBklcjA/Pz8tI6I+vj4ICYmBpcuXUJ0dLQq/eTJk5DJZBg2bBg8PDwAAGFhYdiwYQOOHDmC0aNHQxBMM1cmOzsbu3fvxpNPPgmZTIbDhw8jJSUFERERJmkPERERaTqTJaJ1IwGSh/5eaO8j4IdzIpLuAC08tZ978175qrzanjlt7yNg15Xqh0AzC0V46BAXyhQi1t/0NOh2NQ/KyMgAwp2MUxnpBUdWiUzEwcEBACCTyVRpZWVluH79Onx8fFSBKgBYWloiIiICd+/eRXZ2ttHbWuHnn3+GQqHAmDFj8MorryAnJwfr16/XmjcnJwejRo2Co6MjbGxs0KxZM/zvf/9Dp06d4OHhgUuXLqnlP3HiBHr16gUnJydYWlrCx8cHzz//PDIzjbDiAhER0SMkoxDwsddMr0hLL6g8OswoVM/78Pm5JahyZPZQmohj6cCY8Oo/WN9wSURuKcMRqhxHVomMRC6Xq14Vz6QCQEBAgCpPbm4uFAoFGjVqpHG+l5cXgPLRzYqvK1NWVgaFQqFTuywsLGBpaalT3h9++AGhoaHo1q0bvLy84Obmhh07duDNN99Uew6ktLQUXbt2RWJiIrp3747OnTsjOTkZkyZNUgXpD9q+fTuGDx8OT09PPP300/D19cXp06exevVqXLhwAbt374anZyUfARMRET3ClKIImW6/0mFtAQiCgGJ5+dcPs/n3L/9ieeVlVBzTer7F/TzWWqKIrEIRY3cq0NgZmNG++iD0i3+UkECEEsaZMebv72+Uekh/+FEGkZFcvHgRv/zyC3799Vfs2rULMpkMPXv2hI+PjypPYWH5x5n29pofZ1akVeSpypEjR/DLL7/o9Dpy5IhO7T9x4gQuX76sarNUKsXYsWMRFxeHU6dOqeVdsWIFEhMTMXLkSOzZswcLFizA+vXrsWDBAo2R0pKSEowfPx7BwcE4cuQIfvjhB3zwwQf4/fff8emnnyI2NhbLly/XqY1AecBfWlqqel9QUIB79+6p3stkMuTkqD9Ik5GRUeX7zMxMiOL9T5FZB+tgHayDdbAOY9Wx5WwObL9S6PS6lFteh61URKlCs46SfwPRsuJ7anU82G7bf4PQ0gcC5IrrKPk3rfjeHY3ryMy9hye3KHBPBmx8UonSe7lV9k16egZyS2C051UBICsryyy+5w2lDl0I4oOtICKDKSgoQF5eHuRyOW7fvo3r168jLCwMLVq0UOVJSkrCgQMH0K1bN43nQPPz8/Hbb7+hefPm6Ny5c5V13blzR6egFigPgl1dXavNN378eKxduxZ//vknunXrBgA4e/YsoqOj8dxzz2HVqlWQSMo//+rduzcOHDiAmJgY9OzZU1WGTCaDu7s7rK2tceTIEYSHh2P79u0YMmQIJk2ahNmzZ6vVmZubi6ioKAwYMAC///47pFJOBiEiooYls1DEH1d1+3N9WKgAZ2sBoSvkCHUVsGuE+vDoj+eVeHmPEueet0ALT+2jmTfvifBfpsCn3SQao6PP7lJg1xUROZPVfx/LFCIGb1Hi71QRe0ZaoHuAbiOlS04r8fY+BUQjjaz+1DYFz/fgPqvmhH/5ERmJg4ODagpscHAwGjdujC1btkAul6NVq1YAoArGlErNzxkrpvXqErC5urrqFIDqqrCwEBs2bEBYWBgEQcCZM2dUxwIDA7F//37cuHEDwcHBAIBr167B2dlZbdQYAKysrBAYGIhbt26p0hITEwEAS5cuxdKlS7XWn5+fj5KSEq1TiImIiB5l3vYCXmhes2Au2kvAoTQRSlFUW2TpRIYIOykQVsWfCH6OAjxtgX8yNQPkkxkior3U26IURTy3S4m910WsHyzROVAFgBeaCZj5t4gShXGCVQuLqrfsofqHwSqRibi7u8PDwwMJCQmqYLWqqb5VTRF+mEwmg1xexQMpD5BKpbCysqoyz/r161FYWIjz58+rRlUftmvXLrz22ms61fmgiskdzz77LAYMGKA1T6NGjWBjY1PjsomIiBqikWECNiaJ2JwkYuS/Cx3dLhKx4ZKIwY8JsJbeDw5T8sp/Dz/mcj9tRJiAny+ISM0XEfDv9jV7ryuRdAeY0kY9sHxjrxLrLolY1leC4WE1e8LQyVrAMJ9crL/pAYUR5no+9thjhq+E9IrBKpEJyeVytfn/bm5usLCwUBt5rJCVlQUAOi00dPToUSQlJenUhrCwMPTo0aPKPMuWLYOLiwumT5+usaF2WVkZpk2bho0bN+KFF16AnZ0dgoKCcO3aNWRkZKhNZy4rK8ONGzdgbW2tSgsNDQUASCQSPP300ybbloeIiOhRMTJMQEcfYPwfSiTkAB62wPdxSihEYF4X9YCy9/rymVvXJt4PC2Z1kGBDkgI91yvwVmsJCsqAxaeUaOEBjH9glPerWCW+jxPRyRewswTWJKjPDBvWRIC9VdW/18eHF2OtkbaRT05ORiffUONURnrBYJXIwIqKimBnZ6eRnp6ejjt37qhNlbW0tERgYCCuXbuGnJwcuLu7AygP8i5evAhnZ2edgtWoqCg0adJEp/ZVN1KblJSk2lZm6tSpWjfUXr9+PU6ePIkLFy6gXbt2GD58OPbv34/vv/8enTt3VgWnS5cuRUFBgVqw2q9fP7i5uWH79u04dOiQxshtcXEx8vPzta6QTERERJosJOXPq77ztxJLzihRXAa08wZ+GmCBcLfqPxQOcBLw9xgLTD2gxLsHlbCyAAaFCPi8h0RtVDYuq3w49Fg6cCxd8xGmqxMsYF/15C10b+aHfilAzHUYfHRV22NWVL9xgSUiA/vzzz9RVFQEX19fODg4QKFQ4Pbt20hJSYFUKsWTTz6ptqfq3bt3sXXrVkgkErRo0QKWlpa4ePEicnNz0b9/f7WtbozhnXfewWeffYYFCxbgv//9r9Y833zzDd588028+eab+Prrr1FaWoqoqChcunRJbeua3bt3w9HRETKZDEePHkVYWBgAYPfu3Rg6dCisrKzQv39/REZGoqioCElJSdi3bx9mzZqFWbNmGfOyiYiIyAji4uKQ5dIS/TYaPpDc3Csbw1r7VJ+R6g0Gq0QGlpKSgsuXLyMnJwclJSUAyhdb8vf3R1RUlNZFg+7cuYOTJ08iIyMDCoUCHh4eaNOmjdH3B1MoFPDx8UFJSQkOHz6Mli1bas138+ZNBAQEICIiAvv27YO3tzeys7MxadIk7NmzB2VlZWjSpAlmz56NhQsXIicnB4cOHVJ7duTcuXOYNWsWjh07hrt378Le3h6NGjVCly5dMHXqVLVVk4mIiOjREBcXh6ioKIT9qEBynmHr+uupEvQJ5WKN5oTBKhEZjUKhgKurKwICArBv3z5O7SUiImrgsrKy4OXlhe/PKDF5rxKGDEy4dY35qdmSXUREOiooKNB4NuS7777DvXv30KJFC7Wpz0RERNQwVSys+FwzAfaWJm4M1TtcYImIDOLFF1/E9evX0alTJzg6OiI2NhZ//PEHvLy8MG3aNO51RkRERLh58yY8PT3hYCVgQksBS06LBltoydfX1zAFk8FwZJWIDKJfv37Iz8/H8uXLsXDhQhw/fhx9+vTB77//jnbt2pm6eURERFTPTG4lgdKA84C17WNP9RufWSUiIiIiIpMoLS1V29Ju8GY5dl81zDY2fGbV/HBklYiIiIiITCI1NVXt/dttJAabBlzxfCyZDwarRERERERkEgUFBWrvewUKCHcFDBFWhoaGGqBUMiQGq0REREREZBI2NjZq7wVBwNS2hglRUlJSDFIuGQ6DVSIiIiIiMokmTZpopI2LFOBopf+6FAqF/gslg2KwSkREREREJhEfH6+RZm8l4JUoARZ6ngvs6Oio3wLJ4BisEhERERFRvfJ6tAT63rPE39NZvwWSwUlN3QAiIiIiImqYvL29taYHOQvYPlyCvNLqy8jNvQM3N9cq8/jaA1a5l4HA6Fq0kkyFwSoREREREZlEZcEqAAwM0W0SaF6eBVxcqs8bl6tzs6ie4DRgIiIiIiIyW9euXdMpX3BwsEHbQfrHYJWIiIiIiB55hYWFpm4C1RCDVSIiIiIiMlvatr/RJjs728AtIX1jsEpERERERGbr9u3bpm4CGQiDVSIiIiIiMlt5eXk65YuKijJsQ0jvGKwSEREREZHZkkp12+AkMTHRwC0hfePWNUREREREZLaaN2+uUz6ZTKY1/XaRiBG/K6AUa1Zve28Bn/WQQBCEmp1IOmOwSkREREREZuvs2bM6TfF1cnLSmr7ivIjDaYCyhvUevinio8cBO8sankg64zRgIiIiIiIyW6Ko25Cot7e3RlqZQsTXp5U1DlTJOBisEhERERGR2XJ3d9cpX1JSkkba1mQRmdx+td5isEpERERERGarsum9uvjyHyUkfOS03mKwSkREREREZuvq1as65QsMDFR7f/qWiGMZqPHCSmQ8DFaJiIiIiOiRV1paqvb+61glpBxVrdcYrBIRERERkdkKCQnRKd+tW7dUX2cVivj1ogg5R1XrNQarRERERERktvLy8mp8zg/nRE7/NQMMVomIiIiIyGzl5ubqlK9FixYAyrerWXJayWDVDDBYJSIiIiIisyWR6BbSXL58GQCwMUlEdrEhW0T6wmCViIiIiIjMVsuWLXXKV1JSAgD4gtvVmA0Gq0REREREVGt5JSIm/qmA53dy2H8lR891Cpy+pfsc28QcEf03KuDwtRxu38rx7C4Fsos0z//4uBJDtijQ6Hs5hM/k+OCIAgBw/vx5nepxdHTEyQwR/9zidjXmol4Eq//88w9++OEH3Lt3z9RNMVuNGjVC06ZNjV5vdHQ0PDw8jFLXl19+CUEQ8MsvvxilPiIiIiKqmlIUMWizAr8mipjcSoJF3SXIKhLRY50Cl+9UHxGm3RPR7TcFkvNELOgqwfS2Euy8IqLvBgVkCvXz3zusxKlMEa281IdFFQqFTm318/PDktPcrsacSGtzUnp6Onbs2FHp8aeeegqNGjWqdaPM3eTJk/Hdd99Verx58+Y6fwJEhpGYmIj33nsPR48eRU5ODqRSKQICAjBkyBDMmTMHjo6Opm6iUeTl5WHGjBno1asXnn76aVM3h4iIiMzMxksijqYDGwZLMDK8fBxsdLiAsB8VmHtEiV+ftKjy/AXHlSgsA2KftUCgU3kU2d4H6LtBiZ/iRUyMuh9ZXp1ggWBnAbeLRHh+fz9AdXV11amth84mY93FCG5XY0ZqFaxWeOyxxxAYGKiR7uzsXJdiHxkvv/wywsPDNdJ9fHxM0Bqq8PPPP2PixIkQBAEDBgxAdHQ0iouLERMTg88++wybNm3Cn3/+iSZNmpi6qQaXl5eH5cuXQyaTMVglIiKiGtuYJKKRHTA87H5Q6WknYHS4gDUJIkrlIqyrGMrcdFnEkyGCKlAFgD5BEoS5KrH+koiJUffzBjtrL8fd3V2ntm5Kc4NSp5xUX9QpWPXw8EBoaKi+2vLIGTRoEIYOHWrqZuidKIq4c+cO3NzcTN2UGjt16hQmTJgABwcH/PHHH2jfvr3a8U8//RTvvvsuhg4dijNnzsDS0tJELSUiIiKq/85kiWjdSIBEUA8k2/sI+OGciKQ7QAtP7efevCciqwho660ZhLb3EbDrim5DoMnJyYiOjq4yT6lcxPp0Dz6ramYM9sxqVlYWDhw4gHXr1uHHH3/EqlWrsG3bNly9elWn80tKSnD06FGsXbsWP/74I37++Wds3rwZZ8+e1cibkpKCbdu2YdWqVfjxxx+xZcsWXLlyRd+XZBAVz2H+9NNPmDx5Mjw9PWFlZYXQ0FBs2bIFALB9+3Y0a9YM1tbWcHFxwYQJE6BUav9c6ODBg4iOjoaNjQ0cHBzQt29fpKSkqOXJz8/Hyy+/jLCwMDg6OsLS0hKNGjXCuHHjNPap2rVrFwRBwNy5c/HBBx8gICAAVlZWmDp1aqXXdOvWLURGRsLOzg4///yzKn3jxo1o06YN7OzsYGlpiaCgIMyaNUvrtSxcuBB+fn6qtk2dOrXSa/7nn39w5syZStvzoBkzZqCsrAwff/yxRqAKADNnzkSvXr1w4cIFLF++XJX+5JNPQhAErc9VC4KA3r17a6R///33aNasGWxtbWFtbY3Q0FB8+eWXWtula99UPJt8+vRpdOjQAba2trCzs8Pjjz+OS5cuqeXNysrC2LFj0ahRI1haWsLBwQEhISGYPHkygPLvbePGjQGUjzYLggBBENSeQZbL5Zg2bRoCAgJUZXTq1An79+/XaFdkZKRa2vTp0yEIAtq1a6eW/tJLL0EQBMTHxwO4/3/g559/xowZM+Dt7Q1LS0v4+Pjgww8/1NpfREREVD9kFAI+9prpFWnpBZVHhxmF6nkfPj+3pDzI1IcNSSLulNaL5XqoBuo0siqXy1VLQFeQSCSwsrLCtWvXkJeXh5CQEDg4OKC0tBRJSUn466+/0KtXr2qnWMbExCAjIwNNmzaFm5sbFAoF7ty5g/T0dERF3Z8PcOrUKZw5cwYBAQFo27YtBEHA1atXERMTgy5duqBZs2bVXsfD11AVKysrnfdyysvLQ1pamka6k5MTnJyc1NLmz58PhUKB5557DgqFAitXrsTYsWPx9ddfY+rUqRg2bBhGjBiBbdu2YcWKFQgKCsJ7772nVkZOTg4GDBiAbt26YfDgwYiLi8POnTvRvXt3xMXFqYKQGzduYMOGDejVqxeGDRsGS0tLHDlyBGvXrsWFCxcQGxsLCwv15wt++eUX5OfnY8SIEfDz80NAQIDWa7548SJ69+6NwsJCbNmyBf369QMAfPLJJ5g1axZCQ0MxceJEODo64sCBA1i4cCEuX76MDRs2qMp4//33MX/+fAQFBeGNN95ASUkJfv7550qnl7dr1w7u7u64fft2ld+PkpISHDlyBK6urpgwYUKl+d544w3s27cPW7ZswWuvvVZlmZWZNGkSli1bhtatW2Py5MmQSqXYtWsXpk6dips3b+Kzzz5T5a1J3wDlG1/36tUL3bp1w4wZM3D+/Hls3boVo0ePxpkzZ1T354ABAxAXF4chQ4YgOjoaJSUluHjxIo4dOwagfJn3WbNmYcGCBejYsSNGjBgBALCzs1PV1b9/f+zduxdRUVEYN24csrKysG7dOgwYMACbN2/GwIEDAQCdOnXCzp07cevWLdXz6vv374cgCLhw4QIKCwthb1/+m+jw4cPw9fXVWBBswYIFKCkpwZgxY2Bra4s1a9Zg7ty5CA8Px5gxY2r1fSAiIiLdKUURMt3WKoK1RfkH9sXy8q8fZvNvlFEsr7yMimNaz7e4n8e6moglODi4yuOiKOLzU0oIECGCqyuZkzoFq7GxsYiNjVVLCwkJQZ8+fdCqVSuNkavmzZtj06ZNOH36dJXBqkwmQ3p6Opo2bYouXbpUmu/27ds4c+YMoqOj1epq3rw59uzZg5MnTyI0NBRWVlZVXkdNVpd98skn4evrq1Pe8ePHa01/7bXXNBZgksvliI+Ph4ODAwCgbdu2ePbZZ/Hqq69i8+bNeOqppwAA7733Hho1aoSffvpJI1jNysrCm2++ia+//lqV9t577+Hjjz/G7NmzsWzZMgBAaGgoMjMzYWtrq3b+q6++iqVLl2Lbtm0YPny4RtlnzpxBWFhYpdd7+PBhDB48GLa2tti3bx9at24NAEhNTcWcOXPQtWtX7N+/Xy3YHzduHNauXYtjx46hU6dOyM3NxeLFi+Hr64vTp0+rphpPnz69zqsdJyQkoKysDCEhIZBKK7/1u3XrBqB8SkltHDlyBMuWLcMzzzyDX3/9VZW+cOFCdOvWDd999x2mTJkCPz+/GvVNhVu3buHTTz/FjBkzVGljxozB+vXrcfDgQfTo0QN3797F6dOnMWjQINUI/cP8/f0xYcIELFiwAOHh4Zg+fbra8S1btmDv3r3o0qULDhw4oOqziRMnokuXLpg6dSr69+8PiUSCvn37Ytu2bdixYwdeeukllJaWIj4+Hr1790ZMTAz27NmD4cOHIy8vD8nJyRg0aJDGhz6lpaW4cOGC6v/Aq6++iiZNmuDbb79lsEpERGQEB1NF9Fyv21OdieMtEOEO2EqBUi0Bbsm/gahtFdFGxTGt5yuqP79CQUEBXFxcKj1+Ix+IywbAQNXs1GksPCIiAgMHDlR7VQQoDz7rVzECK5fL4evri7y8PMhkskrLtbCwgIWFBbKysqrczuby5csAgLCwMJSUlKi9goKCUFZWhqysrGqv4+FrqOql6wPcQPmU019//VXjNWnSJI28Y8eOVf2RDpQ/7woATZo0UQWqQPnIbvPmzZGZmYmysjK1MmxsbDBv3jy1tPfffx+2traIiYlRpVlbW6sCVZlMhszMTKSlpWHw4MEAyoPOhz3xxBNVBqqbNm1C37594enpiaNHj6ruAwBYuXIlysrKMH78eKSnpyMtLU31GjVqFERRxO+//64qp7S0FM8884zaM7EhISGqPnmYKIrVjqoCUE1xrm6l34pR7+Li4mrL1OaHH36AIAiYNGmS2rWmpaVhyJAhKCkpwe7duwHUrG8quLi4aEzDrhjBvnDhAgDA1tYWlpaWuHDhAuLi4mp1HWvXrgUAzJ49Wy2479ixo2raccXU44pnsyvus71790Imk+Gdd96BnZ0d/vjjDwDAjh07oFQq0bNnT436Hv4/EBQUBH9/f62zE6qSm5uL0tJS1fuCggK1nyMymQw5OTlq52RkZFT5PjMzE6J4fxoS62AdrIN1sA7W8SjWEeEuYFV/Cb7smI9V/SWq11ed7mFlP0H1/puuxXCzLP9b3sceSLsr16jj8q0CAICvg1DpdVRM/80o1LyOa7mlcLO5vzjTw9cBAEX//q1W8XdgZX0V6AS08AAk0P8DqxmZmWrvze17bso6dFGnkVVnZ2f4+/trPVZcXIxTp07h+vXrWv/ol8lklY54WlhYoFOnTqpnVl1dXeHr64vg4GD4+fmp8uXl5QEA1q9fX2kbi4qKqr2Oyq6hrjp16qTzAksPL1RVsQS3tlFcZ2dnFBYWoqSkRO1DAS8vL41PlaytreHr64tbt26hrKxMlX/evHlYtWoVbty4oXYjAsCdO3c06qxqJDw/Px9jxoyBn58fjh8/rrHwUkUAVdlIMwDVhwpJSUkAoHX6dvPmzbF58+ZKy6hORbuq2883Pz8fAKr8hK4qSUlJEEUR3bt3rzRPxX/WmvRNBS8vL42RYW9vbwD3f1hbWVmpnjNu1aoV/Pz80L59e4wYMQJjx46FIFT/yeKNGzcgCILWZ3sjIyNx8OBBJCYmIjIyUjU1/OTJkwDKg1InJyf07NkTrVu3Vk09/uOPPyAIAoYMGaJRprbF2pydnWscrD58/z0YAAPlffPwh04Pr9D98PuK/mUdrIN1sA7WwToe5Tq87QW80FwA8PAimg9vDXP/g/9oLwGH0ixg/0A9VlZWuHDPAnZSEWGuVV+Hpy3wT6aIGe3Vr+NsrhWiH9hP9eHrAAC7h2YJVlaHIAiY2laC8X/ofy1gn4f639y+56asQxd1ClYrI4oidu3ahTt37qB58+aqRYMEQUBSUhKSk5M1AqSHNW3aFMHBwbhx4wbS09Nx5coVXLhwQTXN+EEDBgyo9I9vXVas1SWgrWBtba3xPKc+VFamrs/H1sTs2bOxYMECREdH48UXX0RgYCCsra2RkZGBadOmaf3ePPgc48Ps7e3RtGlTHDt2DJ9//jk+/vhjteMV5c2fPx8hISFay4iIiKjDFemmadOmsLS0xJUrVyCXyyudCnzo0CEAUHsut7Lvg1yu+SCGKIoQBAHLly+vtN/atGmjygvUrG+quv8eXJBp9uzZGDt2LH799Vf8/fff2L9/P7Zs2YKlS5fiwIEDer+PO3XqhI0bN+LatWs4dOgQoqKiYGlpiV69emHhwoXIycnBkSNHEBgYqPVaDfH/ioiIiAxrZJiAjUkiNieJGBle/vf47SIRGy6JGPyYoLZtTUpe+d89j7ncTxsRJuDnCyJS80UE/Lt9zd7rSiTdAaa00W3abnUrAQPA0xEC3oqRI19ukPCHDMQg363c3Fzk5OSgdevWaNu2rdqxixcv6lyOnZ0dIiIiEBERAaVSif379yMlJQVZWVnw8vJSTdd0cHDQeTNgbdasWaNz3po8s2psWVlZyMvLUxsRLC0tRXp6Onx8fFSjquvWrYOHhwdOnDihNrq9cePGWtVrYWGB/fv3o2/fvli4cCFkMhkWL16sOl4xfdjDwwPPPPNMlWVV5K0YcXxQxeqxtWVjY6N6/vLHH3/EK6+8ojXft99+C6D8mdEKFfdXZmam2jTiipHgB4WEhODEiRMIDg7Wukrwg2rSN7XRuHFjzJ49G7Nnz4ZcLsegQYPw559/Yvv27Rg6dGiVI6xBQUE4ceIETp48iQEDBqgdq/h//OAKwP369cP69euxbt06JCYm4t133wVQPkX4ww8/xOrVq3H9+nWMGDFCp5FdIiIiqv9Ghgno6AOM/0OJhBzAwxb4Pk4JhQjM66L+YX/v9eUPol6beD8EmdVBgg1JCvRcr8BbrSUoKAMWn1KihQcwvrn63wurLyhxPR8o+neF4INpwPxjSmRl3cK0Ht4IqmQfVgCwkQr4T8g9LLvsCgW3rzEbBlm/ubI/RHNzc3Ht2rVqz5fL5RojVhKJRDW0XDFfumLa4KlTp7Rua6LriKmhnlk1tpKSEsydO1ct7aOPPkJxcbFa0FQxSvhgn5WVlWH+/Pm1rtvKygp79+5Fz5498dlnn+Htt99WHXv55ZdhaWmJRYsWaZ1inJOTo5qaO3z4cFhZWWHt2rVq2+hcuXIFu3bt0lp3TbauWbRoESwtLTF79mycPn1a4/hnn32GmJgYtGjRQi1YrRjdfHixovfff1+jjIogeNasWWpz+yukpqZCoSj/YV2TvqmJgoICjfKkUilatmwJ4P7U4oopHNrqfvrppwGULwxV0V6g/P/boUOHEB4ejvDwcFX6U089BUEQ8M0330ChUODJJ58EALRq1QrOzs74/PPPIYqixswIIiIiMl8WEgG7RlhgTISAJWeUeOdvJTxsgX2jLRDuVv2H0wFOAv4eY4HHXAS8e1CJRSeVGNhYwF+jLNRGZQHgx/NKvH9EiYUnyqPN/aki3j+ixDeXPXH1bvUR6PTHte8sQfWXQUZWXVxc4OrqirNnz0Iul8PZ2Rl3795FYmIi3Nzcql0MJy8vD9u3b0fjxo3h6uoKa2tr5OXlISEhAY6Ojqr5zl5eXmjTpg1iY2OxadMmhISEwN7eHkVFRcjOzkZqaipefvnlattrqGdWd+7cqXVFWVtbW7z22mt6H13y8vLCihUrkJSUhLZt26q2rvHx8VGbmvvUU0/hs88+Q8eOHTFkyBDcu3cP27Ztq/OUY6lUij///BMDBw7E119/Dblcjm+//RaNGzfGwoUL8c477yAsLAyDBw9G48aNkZ2djYSEBBw6dAgHDx5Ehw4d4O7ujmnTpmHhwoVo3bo1RowYgdLSUqxduxY+Pj5a9+nVdeuairzLli3DpEmT0LlzZwwcOBCtWrVCcXExYmJicOrUKYSEhGD79u1qzwO/+uqrWLBgAT744AMkJyfD09MTe/fu1XiQHAC6d++OyZMn49tvv0WTJk0wcOBA+Pv7IyMjA2fPnsWJEydw584dODo61qhvaiIhIQE9evTA448/jubNm8PNzQ0XL17Ehg0b4OnpiWHDhgEA3N3d4ePjg/3792PGjBnw9fWFra0tXnnlFQwbNgy9e/fG3r170a5dO/Tv3x9ZWVlYu3YtLC0t8cUXX6jdM+7u7ggJCUFKSgo8PDzU9ldt164dYmJiYGFhUelCWURERGSeXG0ErOhngRX9qs734Ijqg5p5CNgzsvrHgQ48rf38q1evonFg42rPv3sjASPDWmBTkgg9bd9KBmaQYFUikaB///44fvw4kpKSUFZWBjc3N/To0QM5OTnVBhUODg4IDw9HRkYGrl27BoVCAXt7e0RERCA6OlrtWcM2bdrA09MT8fHxiI+PR1lZGWxtbeHm5obOnTsb4vJ0tmLFCq3pLi4uePXVV/UerLq7u2PDhg148803sWjRIkilUvTs2RNLly6Fp6enKt8nn3wCuVyOtWvXYuHChXB2dkbfvn0xc+ZMtGrVqk5tsLCwwO7duzFkyBB89913KCsrw9KlSzFt2jQ0bdoUH3/8MTZu3IiioiI4OjrC398fr732mtoI3YIFC2Bvb4/vv/8eS5YsgZubG55//nn4+flpbK9SG+PHj0f79u3x/vvv4+jRo9i+fbtqJL9Xr17YtWsXrK2t1c5xcXHBli1b8Oabb2LVqlWwsbFBt27dsGnTJq0fdnzzzTfo0KEDvvrqK/zvf/9DaWkpnJycEBwcjHfffVdt26Ca9I2ugoOD8dRTT+H48eM4fPgw5HI53NzcMHjwYHz88cdq98OaNWswefJkfP3115DJZHB3d1eNDv/xxx+YOXMm1q1bh8WLF8Pa2hotW7bExx9/jF69emnU26VLF6SkpKBVq1Zqz6D27dsXMTExCAkJMdiHQ0RERNQwVezxrou32kiw7pKOm8mSyQlidSsdETUAN27cQMeOHZGTk4PVq1dj9OjRpm4SEREREekgLi5Op0WWsrKy4OnpiTarFTibDSj1EAUVvmUBO0uuxWEoBnlmlcjcBAYG4tChQ3BxccELL7yAbdu2mbpJRERERKRHgiCotrHRR6BKhseRVSIiIiIiMlu5ubk6bVdZMQJbKhfht1SBnJK6182RVcPiyCoREREREZmtkpKaRZ3WUgGTWwmQMMas9xisEhERERGR2arYkq86D+4PPylawkDIDPB7REREREREj7zU1FTV1972AsZECJBydLVeY7BKRERERERmq0WLFjrlKygoUHv/VmsJ91ut5xisEhERERGR2UpKStIpn42Njdr7dj4C2jYCn12txxisEhERERGR2SotLdUpX5MmTTTSprXjNjb1GYNVIiIiIiIyW46Ojjrli4+P10gbESrAy07fLSJ9YbBKRERERERmy8/Pr9bnWloIeKOVhFOB6ympqRtARERERERUWxcvXkR0dHS1+by9vbWmT2wpYHsKIFfWrN7WHnLYSC1qdhLVCINVIiIiIiJ65FUWrHrZCzjxn5qHRXFx8ZAI0XVsFVWF04CJiIiIiMhs+fv7m7oJZCAMVomIiIiIyGzJ5XKT1BscHGySehsSBqtERERERGS2MjMzTVJvYWGhSeptSBisEhERERER1VB2drapm/DIY7BKRERERERmq3nz5qZuAhmIIIqiaOpGEBERERER1calS5cQHh5u9HpFUYQgcINWQ+LWNUREREREZLaKi4urz5R7D9gfX7OC7ayBAa0rPZyYmIimTZvWrEyqEQarRERERERktuzt7avPNOF7YPOJmheesASI1L41jkwmq3l5VCN8ZpWIiIiIiMxWYGBg9ZlyC2pesFQCLNlZ6WEnJ6eal0k1wmCViIiIiIjMVmJiomEKliuBn/YBedq3qPH29jZMvaTCYJWIiIiIiEibUjmwcq/WQ0lJSUZuTMPDYJWIiIiIiMyWr6+v4QoXReDL7YBCYbg6qFIMVomIiIiIiCqTlgPsiNVI1ulZWaoTBqtERERERGS20tPTDVuBhaR8dPUhpaWlhq2XGKwSERERERFVSqEE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7xYjIiJEKysrEYDo4eEhimLVKwVXrG774JY2oiiK8fHx4pAhQ0QPDw/RwsJCtLe3F0NCQsTnnntOTExM1LjOmqiqPaIoiitXrhRbt24t2tvbi1KpVHRzcxPbtGkjzpkzRy3f5s2bxcjISNHa2lq0tbUV27VrJx4+fFhs3ry56torXLt2TezSpYtob28vCoKgcc1vv/226O7uLkqlUtHX11ecP39+lasBP9xfFUpLS8V3331XfOyxx0QrKyvRyspK9Pb2Fnv37q22kvEbb7whNm3aVHR0dFRdY9euXcWYmJga9SURERE1LDqtBnwjW3s+qtcEUXxoziAREREREZGZiIuL0xwxjZ4KnL1W/rWnE3BrFbKys+Hl5WXs5lEd8JlVIiIiIiIyW1VuWSiRAI9HAoKgt8U3yXgYrBIRERERkdl6eG95DZ3CAajvTU/mgcEqERERERGZrVu3blV+UKks31aGzBKDVSIiIiIiejRJBKDNYwCAyMhIEzeGaorBKhERERERma0WLVpUfrBpAGBnDQBITU01UotIXxisEhERERGR2bp8+XLlB7veH00tKCgwQmtInxisEhERERGR2SopKdFMzC8q//eB51VtbGyM1CLSFwarRERERERkthwdHTUTr2aV/9sxXJXUpEkTI7WI9IXBKhERERERmS0/P7/KD4b6qL6Mj483QmtInxisEhERERGR2bp48WLlBwXBeA0hvZOaugFERERERER69eccXDt9AcEPJHl7e5uqNVRLgiiKoqkbQUREREREVBu3b9+Gh4eHzulkPjgNmIiIiIiIzJZCoahROpkPBqtERERERGS2MjIyapRO5oPBKhEREREREdU7fGaViIiIiIjMVllZGSwtLXVOJ/PBkVUiIiIiIjJbV69erVE6mQ8Gq0REREREZLaKiopqlE7mg8EqERERERGZLXt7+xqlk/ngM6tERERERGS2ZDIZrKysdE4n88GRVSIiIiIiMlsJCQk1SifzwWCViIiIiIiI6h0Gq0REREREZLZ8fX1rlE7mg8EqERERERGZLUEQapRO5oPBKhERERERma2bN2/WKJ3MB4NVIiIiIiIiqne4dQ0REREREZmt0tJSWFtb65xO5oMjq0REREREZLZSU1NrlE7mg8EqERERERGZrYKCghqlk/lgsEpERERERGbLxsamRulkPvjMKhERERERmS25XA6pVKpzOpkPjqwSEREREZHZio+Pr1E6mQ9+1EBEjwyFQoGkpCRTN4OIiIiM6MqVK1pX/a0sneqHsLAwWFhYVJmHwSoRPTKSkpLQtGlTUzeDiIiIiKqRkJCAyMjIKvPwmVUiemQYc2S1oKAA7du3x8mTJ+Hg4GCUOhsK9q1hsF8Nh31rGOxXw2go/ZqZmYlevXph37598Pb2NkqdDaVv9UWXkVUGq0REtZCfnw9nZ2fcvXsXTk5Opm7OI4V9axjsV8Nh3xoG+9UwGkq/pqWlISAgAKmpqfD39zdKnQ2lb42JCywRERERERFRvcNglYiIiIiIiOodBqtERLVgbW2NuXPncpVBA2DfGgb71XDYt4bBfjWMhtKvTk5O6N69u1Gn4zaUvjUmPrNKRERERERE9Q5HVomIiIiIiKjeYbBKRERERERE9Q6DVSIiIiIiIqp3GKwSERERERFRvcNglYgaJKVSiS+//BIRERGwsbFBQEAApk2bhsLCQp3OX7hwIUaNGoWQkBAIgoDg4OAq8584cQJ9+vSBo6MjnJyc0L9/f8TFxdX9QuoZY/brCy+8AEEQtL42btyopyuqH+rSr0lJSZgzZw46duwIT09PODo6Ijo6Gh9//HGl51+6dAlDhw6Fq6sr7O3t0bVrV+zbt0/fl1UvGLNvP/jgg0rv2c8++8wQl2cydenXS5cuYdy4cYiMjISzszPs7OwQERGBqVOnIiMjo9JzeM/qv295z+r+++tBRUVFqt9jkydP1pqnId2zdSE1dQOIiExhypQpWLJkCYYNG4Zp06YhMTERS5YswZkzZxATEwOJpOrP8mbNmgU3Nze0bt0aeXl5VeY9fvw4evToAT8/P3z44YcAgG+//RZdu3bF0aNH0aJFC31dlskZs18rrF69WiOtffv2tWl+vVWXfl25ciW+++47DBkyBOPGjYOlpSX279+P9957D+vXr8fx48dha2uryp+SkoLOnTtDKpVixowZcHZ2xvLly9GvXz/s3r0bffr0McYlG40x+7bCl19+CQ8PD7W0Nm3a6P3aTKku/ZqWloaMjAwMGzYM/v7+kEqlOH/+PH744Qf89ttviIuLg5eXlyo/71nD9W0F3rPV//560Jw5c5CdnV3p8YZ2z9aJSETUwMTHx4uCIIjDhw9XS1+yZIkIQPzf//5XbRkpKSmqr5s1ayYGBQVVmrddu3aio6OjmJaWpkpLS0sTHR0dxb59+9b8AuopY/fr888/LzaEX2N17ddTp06JeXl5GumzZ88WAYjffPONWvqoUaNEiUQinjlzRpV27949MTAwUAwLCxOVSmXtL6aeMXbfzp07VwQgXr16tc5tr8/08bNAm/Xr14sAxE8//VQtnfes4fqW92zN+zU2Nla0sLAQP//8cxGA+Prrr2vkaUj3bF1xGjARNThr166FKIp4++231dInTJgAOzs7rFmzptoyQkJCdKorOTkZp06dwqhRo+Dn56dK9/Pzw6hRoxATE4PMzMwatb++Mma/PkgUReTn50OpVNb4XHNQ135t27YtnJ2dNdLHjBkDAIiPj1elFRYW4vfff0ePHj0QHR2tSndwcMDLL7+MpKQknDp1qvYXU88Ys28flp+fD7lcXvNGmwF9/CzQJigoCABw584dVRrv2XKG6NuH8Z6tnkKhwIQJE9C/f38MHz5ca56Gds/WFYNVImpwTp06BYlEojFV1MbGBtHR0Xr9JVFRVqdOnTSOdezYEaIoIjY2Vm/1mZIx+/VBzs7OcHZ2hq2tLfr27YsTJ04YpB5TMVS/pqWlAQAaNWqkSjt37hxKS0srvV8r2vOoMGbfPqhly5ZwdnaGjY0NOnfujN27d9eqnvpKX/1aUlKC27dvIy0tDX/++SdeeeUVAMDAgQNVeXjPljNE3z6I96xu/frll1/i4sWL+PbbbyvN09Du2bpisEpEDU56ejo8PDxgbW2tcczPzw+3b9+GTCbTW10V5WqrCwBu3rypl7pMzZj9CgDe3t6YMmUK/u///g9btmzBrFmz8M8//6Br166IiYnRWz2mZoh+VSgU+OijjyCVSjF27Fi1uirK1VYX8Ojcr4Bx+xYAXFxcMHHiRHzzzTfYtm0bFi5ciOvXr2PQoEH46aef6nIp9Yq++nXFihXw9PREQEAA+vXrh7y8PKxZswZdu3ZVq6uiXG11AbxntdGlbwHes4Du/Xr16lXMnTsXc+bMqXJxwIZ2z9YVF1gioganqKhI6y8koPxT1Io8VlZWeqkLgNb6HqzrUWDMfgWATz75RO390KFDMXbsWERHR+PVV1/F5cuX9VKPqRmiX99++20cO3YMCxYsQHh4uFpdQMO4XwHj9m3FsYe9+OKLaN68OaZMmYKRI0fCwcFB9wuop/TVr0OHDkVERAQKCgpw5swZ/P7777h9+7ZGXQDvWUD/fQvwngV079dJkyYhJCQEU6dOrbYuoOHcs3XFkVUianDs7OxQWlqq9VhJSYkqj77qAqC1Pn3XZWrG7NfKhIaGYvTo0UhOTkZSUpJB6zIWfffr+++/j2+//RYTJ07Ef//7X426gIZxvwLG7dvKuLu7Y9KkScjLy8PRo0d1rqs+01e/+vv7o0+fPhg6dCjmzZuHn3/+GTNmzMDChQvV6gJ4zwL679vK8J7VtGbNGvz111/4v//7P1haWlZbF9Bw7tm6YrBKRA2Or68vbt++rfUXxc2bN+Hh4aG30T9fX19VudrqArRPBTJHxuzXqlRMv9I2SmCO9NmvH3zwAebPn4/x48dj6dKlWuuqKFdbXcCjc78Cxu3bqvCe1U3Lli3RqlUrfP/992p1VZSrrS6A96wutPVtVXjP3ldaWoqpU6di4MCB8Pb2RnJyMpKTk3H9+nUAwN27d5GcnKzajq2h3bN1xWCViBqcdu3aQalU4uTJk2rpJSUliIuLQ9u2bfVaFwAcO3ZM49jx48chCMIjs1edMfu1KhXTfytb3Mbc6KtfP/jgA8ybNw/PP/88VqxYAUEQNPK0aNEC1tbWld6vAIz2fTQGY/ZtVXjP6q64uBi5ubmq97xnyxmib6vCe/a+4uJiZGdnY+fOnQgNDVW9evToAaB81DU0NBQrVqwA0PDu2Toz2aY5REQmcu7cuSr3U1u9erUqLTk5WUxMTKyyvOr2A23btq3o6Ogo3rx5U5V28+ZN0dHRUezdu3ftLqIeMma/FhQUiMXFxRrpp0+fFq2srMTIyMiaX0A9pY9+nTdvnghAfPbZZ0WFQlFlfSNHjhQlEokYFxenSqvY/y80NPSR2v/PmH1bVlamdU/WGzduiG5ubqK7u7tYVFRUh6upP+rarxkZGVrL3bdvnyiRSMRevXqppfOeNUzf8p7VrV9lMpm4YcMGjdf3338vAhD79+8vbtiwQbx06ZLqnIZ0z9aVIIqiaKpAmYjIVN544w18++23GDZsGAYOHIjExEQsWbIEXbp0wb59+yCRlE88CQ4OxvXr1/Hwj8rVq1erpvh88803kMlkmDZtGoDy/eqeffZZVd6jR4+iZ8+e8Pf3xxtvvKE659atWzhy5AiioqKMcclGYax+jYuLw4ABAzB06FCEhobC3t4eZ8+excqVKyGRSPDnn3/i8ccfN+KVG1Zd+vW7777D5MmTERgYiI8++kiVt0KjRo3Qt29f1fvk5GS0b98elpaWmDJlCpycnLB8+XKcP38eO3fuRL9+/Yxz0UZirL7Ny8tD48aNMXToUERGRsLV1RWXLl3CihUrUFBQgLVr12LUqFHGu3ADq0u/Dhs2DBkZGejVqxeCgoJQUlKC2NhY/Pbbb7Czs8OBAwfU9qfkPWuYvuU9W7PfXw+7du0aGjdujNdff11jK5uGds/WicnCZCIiE5LL5eJnn30mhoWFiVZWVqKvr684ZcoU8d69e2r5goKCRG0/Krt37y4C0Prq3r27Rv6jR4+KvXr1Eu3t7UUHBwfxiSeeEGNjYw11eSZjrH7NyMgQ//Of/4jh4eGio6OjKJVKxYCAAPG5556rdsTWHNWlX59//vlK+7Sy+zUhIUEcMmSI6OzsLNra2opdunQR//rrL0NeoskYq29LSkrEl156SWzevLno4uIiSqVS0dvbWxwxYoR44sQJY1yqUdWlX9etWycOGjRI9Pf3F62trUUbGxsxPDxcnDx5snj9+nWt9fGe1X/f8p6t2e+vh129elUEIL7++utajzeke7YuOLJKRERERERE9Q4XWCIiIiIiIqJ6h8EqERERERER1TsMVomIiIiIiKjeYbBKRERERERE9Q6DVSIiIiIiIqp3GKwSERERERFRvcNglYiIiIiIiOodBqtERERERERU70hN3QAiIiIiIqKHvfDCC/j5558BAM2aNUN8fLzacaVSiQULFmDVqlW4ceMGAgMDkZKSgkWLFmHlypVISEiARFLzsbmlS5diwYIFuHz5MqytrdWOffXVV5gyZYrqfXZ2Njw8PGpxdbWjUCjwzz//4PLlyygtLYWbmxvatWsHf3//as/Nzc1FbGwsbt++jaKiIkilUri6uiIqKgpBQUFqee/evYtTp07h1q1bKCkpgYODA5o0aYKoqChIpdJalVkbHFklIiIiIiKj2rNnDwRBqPT1yy+/AAA8PDywevVqfPLJJxplfP/995gzZw6GDx+OlStXYtmyZcjPz8enn36KmTNnagSq8+bNg0QiQWJiokZZL774IiwsLLBz50688MILkMlkWLZsmUa+/v37Y/Xq1Rg2bJieeqJmDhw4gHPnzqFJkybo3LkzJBIJdu/ejczMzGrPLSgoQFlZGcLCwtC5c2e0bt0aQPn34sE+KSgowJYtW5CVlYVmzZqhc+fOaNSoEWJjY7F3795alVlbgiiKYp1LISIiIiIi0tGiRYswc+ZMLFmyBK6urhrH+/Xrh3feeQcHDhzAtWvXtJbRpk0beHh4YM+ePaq0r776CnPnzsWtW7dgY2Ojlj8rKwtBQUF47rnn1ALRb775Bm+++Sbmz5+P2bNnAwBmzpyJdevW4erVqxAEQaPuDz74APPmzTPqyGpWVha2bt2KDh06ICoqCgAgl8uxceNG2Nra4qmnnqpxmUqlElu2bIFcLseYMWMAAGfOnMGpU6cwcuRIuLm5qfLu378fly9fxvPPP68x4lxdmbXFacBERERERGRU586dg7OzMyZPnqw1GKxOSUkJzp49i3nz5qmlr1q1CkOGDNEIVAHAy8sL48aNw+rVq/Hxxx/Dw8MDf//9N6ZOnYoRI0aoAlUAGD16NBYtWoT9+/ejV69eNb9AA7hy5QoEQUBkZKQqTSqVIjw8HKdOnUJBQQEcHBxqVKZEIoG9vT2ys7NVaTKZDABgZ2enltfOzg6CIFQ7tVpbmbXFacBERERERGRUZ8+eRatWrWoVqL700kuwtbWFQqHAe++9B0EQ0KlTJ1y9ehXnzp1Dnz59Kj13ypQpKC4uxtKlS5GamorRo0cjIiICP/30k1q+Nm3awM3NDdu2batx+7RRKpUoKSnR6VXZxNecnBw4OzvDyspKLd3Ly0t1XBdlZWUoKSlBfn4+zp07h9TUVPj5+amO+/r6AgD+/vtv3L59GwUFBUhJSUFCQgKaNWsGS0vLGpdZWxxZJSIiIiIio5HJZLh06RIef/xx3L59W+O4s7Oz1oCowrhx42BpaYlly5bh66+/hpubG4KCgnD06FEAUD03qU2zZs3wxBNP4LvvvsPWrVtRVlaGrVu3ah2RbN26NY4cOVKLK9SUmZmJHTt26JT3mWeegaOjo0Z6UVGRxmgncH8EtLCwUKfyjx8/rnqeVBAEBAcHo0uXLqrjAQEBaNu2Lc6cOYPr16+r0lu1aoV27drVqszaYrBKRERERERGk5CQgLKyMixduhRLly7VOH7p0iWEhYVVen6vXr2wd+9e2NvbY/Lkyappqe+//z4AoHHjxlXWP3XqVPTv3x9ZWVnYtWsXHnvsMa35QkJCsHr1al0vq0ru7u4YOHCgTnltbW21psvlclhYWGikV6QpFAqdym/RogUaN26MoqIiXLlyBaIoapzr6OgIHx8fNG7cGDY2Nrhx4wbOnDkDW1tbNG/evFZl1gaDVSIiIiIiMppz584BAH766SetU0VDQ0N1KqNZs2Zqz0/m5ORAKpVW+9xmxQhgSEgI+vXrV2k+V1dXFBcXVzqiWRPW1tY6bS9TFalUqjUArEjTFshq4+LiAhcXFwBAWFgYdu7ciT179mDo0KEQBAHJyck4ePAgxowZo+rLxo0bQxRFnDx5Ek2aNNF4Jri6MmuLwSoRERERERnN2bNnIZVK8cwzz2g8f1mTMqoKNCsTExOD6dOnIzQ0FJcvX8aff/6JJ554QmveimdH6xJsVVAoFCgtLdUpr42NjdZFjOzs7LRO9S0qKgIA2Nvb16ptISEhOHToEO7evQsXFxckJCTAw8NDI+gPCgpCUlISbt++XW3g/XCZtcVglYiIiIiIjObcuXNo3LhxrQPVvLw8pKamokWLFmrp7u7ukMvluHfvntZnPq9cuYIxY8agVatWiImJQVhYGL788stKg9U7d+7Azs6u0mm5NXHr1q06P7Pq7u6O9PR0yGQytb7LyspSHa8NuVwO4P4qwMXFxVq3plEqlQBQ6QJQVZVZWwxWiYiIiIjIaM6dO4eOHTvW6XwAaNmypVp6REQEAODq1asaxwoKCvDUU0/B0tISW7ZsgbOzM1577TXMmzcPiYmJatvBVLh69arW9NrQxzOrISEhOHfuHBITE1X7rCoUCly6dAleXl6qkVC5XI6CggLY2NioTdctLi7WKFupVOLy5cuwsLBQ7Xfr7OyMtLQ05OXlqY2KpqSkQBAEtb1XdS2zthisEhERERGRUWRmZiIrK0sVWNbG2bNnAWgGq506dQIA/PPPP2rHRFHEs88+i0uXLmH//v2qKayvvfYaPvnkE3z11VdYtmyZRj2nT5/GuHHjat3OB+njmVUvLy+EhITg5MmTKC4uhrOzM5KSknDv3j10795dlS8rKws7duxA69at0bZtW1X6oUOHIJPJ4OPjA3t7exQVFSE5ORl5eXno2LGjagXmqKgopKamYvv27WjWrBmsra1x48YNpKamIiIiQm26sa5l1haDVSIiIiIiMoqKQDM7Oxtr1qzROB4VFaUxvfdh586dg5+fn9oIH1A+8ti8eXPExMTgxRdfVKV/8MEH2Lp1K5YtW6a2nYqnpyf+85//YPXq1ViwYIHaNNrY2Fjk5ubiqaeeqtV1GkqPHj3g4OCAy5cvQyaTwc3NDf3794ePj0+154aEhODSpUtISEhASUkJrKys4OHhgfbt2yM4OFiVz8fHB0899RRiY2Nx4cIFlJaWwtHREe3atVON6Na0zNpisEpEREREREZRMYV31apVWLVqlcbxX375Radg9eFR1Qovvvgi5syZo5qeumXLFnz00UeYNGkSJk6cqJF/ypQp+PHHH7F06VLMnj1blb5hwwYEBgaiV69eNbk8g5NKpejYsWOV06h9fX21XmuTJk3QpEkTnerx8vLCgAEDqs1XkzJrQxB1eUKWiIiIiIjIiF544QXs27cPp0+fhlQq1WlV2bt37yIkJASLFi3CSy+9VKt6S0tLERwcjHfffRdvvfWW2rGSkhIUFBRg0aJFWLx4MbKzs+Hh4VGreqh6mmsiExERERER1QOpqanw9PTE448/rlN+Z2dnzJgxA4sXL1atXltTq1atgqWlJSZNmqRxbOnSpfD09MTixYtrVTbVDEdWiYiIiIio3klISEB6ejoAwMHBoU4rCOtLamoqLl26pHrfvXv3Oi8iRJVjsEpERERERET1DqcBExERERERUb3DYJWIiIiIiIjqHQarREREREREVO8wWCUiIiIiIqJ6h8EqERERERER1TsMVomIiIiIiKjeYbBKRERERERE9Q6DVSIiIiIiIqp3GKwSERERERFRvfP/NuDV72G+qUgAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 800x650 with 3 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                      contribution  abs_contribution  overall_avg   value\n",
      "Sex: male                 0.249830          0.249830     0.647587   False\n",
      "Fare                     -0.101959          0.101959    32.204208  8.6625\n",
      "Ticket Class: Middle     -0.050638          0.050638     0.206510   False\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 25.0%, Overall Average: 64.8%\\nFeature --> Fare = $9, Contribution: -10.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Middle = False, Contribution: -5.1%, Overall Average: 20.7%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_3 = get_sample(534)\n",
    "sample_3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "face1590-9938-4124-80cc-1fc8794fe7a6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 25.0%, Overall Average: 64.8%\\nFeature --> Fare = $9, Contribution: -10.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Middle = False, Contribution: -5.1%, Overall Average: 20.7%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "explanation_3 = \"This passanger did not survive in spite of being a female which improved her chance of survival by 25.0%. It helped that her $9 fare was much lower than the the average fare of $32.2 that everyone else paid. This had a negative contribution of -10.2% to her survival. Her ticket class was 'Middle', which further worsened her chance of survival by -5.1%\"\n",
    "\n",
    "sample_3"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "257eded1-f329-49cc-8ddb-5bae43c1103d",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "073627f0-2b3c-4bc3-99d0-6631251b78e8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                       contribution  abs_contribution  overall_avg  value\n",
      "Sex: male                 -0.152750          0.152750     0.647587   True\n",
      "Embarked: Southampton      0.061513          0.061513     0.722783  False\n",
      "Siblings                   0.030024          0.030024     0.523008      2\n"
     ]
    }
   ],
   "source": [
    "sample_4 = get_sample(301, False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "ecd8fdbd-1fc5-41f0-a13b-aff92c11a374",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = True, Contribution: -15.3%, Overall Average: 64.8%\\nFeature --> Embarked: Southampton = False, Contribution: 6.2%, Overall Average: 72.3%\\nFeature --> Siblings = 2, Contribution: 3.0%, Overall Average: 0.5\\n\\nOutcome: This passenger survived.\\n\\n'"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "d19461dc-2f90-4c3b-9f3c-8ad9c641f2f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_4 = \"This passanger did survive in spite of being a male which reduced his chance of survival by 15.3%. It helped that he did not embark from Southampton and he was traveling with 2 siblings, which improved his probability of survival by 6.2% and 3.0% respectively.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab0d19a8-22fe-4372-a6b9-5cba5ba01860",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "690f310b-f9c1-4bcd-b1be-886d5944852e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg   value\n",
      "Sex: male               -0.129434          0.129434     0.647587    True\n",
      "Fare                    -0.041977          0.041977    32.204208  7.7958\n",
      "Ticket Class: Upper     -0.023089          0.023089     0.242424   False\n"
     ]
    }
   ],
   "source": [
    "sample_5 = get_sample(756, False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "589b46de-65ff-42d2-852c-b45451bb89ca",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = True, Contribution: -12.9%, Overall Average: 64.8%\\nFeature --> Fare = $8, Contribution: -4.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Upper = False, Contribution: -2.3%, Overall Average: 24.2%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "7a999466-2076-4ada-a36a-2f99fe618e8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_5 = \"There were several factors that contributed to this passangers demise: being male contributed -12.9%, and having a lowe ticket fare -- only $8 as compared to the average of $32.2 that everyone else paid -- contributed -4.2%. Also, it didn't help that his ticket class was not 'Upper'. This reduced his chances of survival by -2.3%.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fc7e1497-c21e-4854-aa3a-5e22447b8daa",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "76c7e2cd-8364-4b33-bc31-39e8e5bfdb99",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg     value\n",
      "Sex: male                0.268978          0.268978     0.647587     False\n",
      "Fare                     0.101691          0.101691    32.204208  247.5208\n",
      "Ticket Class: Upper      0.091127          0.091127     0.242424      True\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 26.9%, Overall Average: 64.8%\\nFeature --> Fare = $248, Contribution: 10.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Upper = True, Contribution: 9.1%, Overall Average: 24.2%\\n\\nOutcome: This passenger survived.\\n\\n'"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_6 = get_sample(299, False)\n",
    "sample_6"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "1632e2bc-4958-431a-8d64-4eb08592b94a",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_6 = \"She survived, largely due to the fact that she was a female. This contributed as much as 26.9% to her survival. She also paid a much higher fare of $248 as compared to $32.2 that everyone else paid on average. The higher fare helped her by 10.2%. Lastly, her ticket class was 'Upper', which worked in her favor by increasing her chance of survival by 9.1%.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41152b69-a68b-430d-80b2-a7ea288c99d9",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "3f780585-3f24-4fba-9d9d-28a3c901c528",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg    value\n",
      "Sex: male               -0.161338          0.161338     0.647587     True\n",
      "Ticket Class: Upper      0.056736          0.056736     0.242424     True\n",
      "Fare                     0.045103          0.045103    32.204208  27.7208\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = True, Contribution: -16.1%, Overall Average: 64.8%\\nFeature --> Ticket Class: Upper = True, Contribution: 5.7%, Overall Average: 24.2%\\nFeature --> Fare = $28, Contribution: 4.5%, Overall Average: $32.2\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_7 = get_sample(64, False)\n",
    "sample_7"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b2e219e0-e70d-4c3f-92c2-66586cc9cef5",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_7 = \"This passanger did not survive, mainly due to the fact that he was a male, which reduced his chance of survival by -16.1%. Two other factors, his ticket class (Upper) and low fare ($28), actually improved his chance of survival, by 5.7% and 4.5%, but this was not enough to overcome the negative factors.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9deee7a5-b24d-49c5-84e0-faacb5b622a5",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "b064d376-5ff4-4b4e-bf11-20a98de718ff",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg    value\n",
      "Sex: male                0.253869          0.253869     0.647587    False\n",
      "Fare                     0.101930          0.101930    32.204208  227.525\n",
      "Ticket Class: Upper      0.090677          0.090677     0.242424     True\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 25.4%, Overall Average: 64.8%\\nFeature --> Fare = $228, Contribution: 10.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Upper = True, Contribution: 9.1%, Overall Average: 24.2%\\n\\nOutcome: This passenger survived.\\n\\n'"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_8 = get_sample(700, False)\n",
    "sample_8"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "a08be995-9347-4f52-8dba-18ded383ad75",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_8 = \"The biggest contributor to this passanger's survial was the fact that she was a female. This increased her chance of survival by as much as 25.4%. Her fare was also higher than the overall average: $228 as compared to $32.2 that everyone else paid on average. The fare further improved her chance of survival by 10.2%. It also helped that her ticket class was 'Upper', which caused a further increase of her chance of survival by 9.1%.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "50f8a5ac-6a78-4316-82e4-3c513686b2f9",
   "metadata": {
    "jp-MarkdownHeadingCollapsed": true
   },
   "source": [
    "#### Sample 9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "e29096f7-b171-496f-855c-20a2852bb647",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                     contribution  abs_contribution  overall_avg     value\n",
      "Sex: male                0.268978          0.268978     0.647587     False\n",
      "Fare                     0.101691          0.101691    32.204208  247.5208\n",
      "Ticket Class: Upper      0.091127          0.091127     0.242424      True\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 26.9%, Overall Average: 64.8%\\nFeature --> Fare = $248, Contribution: 10.2%, Overall Average: $32.2\\nFeature --> Ticket Class: Upper = True, Contribution: 9.1%, Overall Average: 24.2%\\n\\nOutcome: This passenger survived.\\n\\n'"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_9 = get_sample(299, False)\n",
    "sample_9"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "a1f92402-540e-41ba-bb00-eee6a84d9783",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_9 = \"This passanger survived mostly because of her gender -- female -- which contributed 26.9% to her survival. Her higher fare ($248 as compared to $32.2 that everyone else paid on average) and her ticket class ('Upper') further contributed positively to her chance of survival by 10.2% and 9.1%, respectively.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad555da8-461f-4725-9921-c05d9398b279",
   "metadata": {},
   "source": [
    "#### Sample 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 119,
   "id": "80dbe1ed-3c94-4a9e-97bc-ecb2618f895c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                       contribution  abs_contribution  overall_avg  value\n",
      "Sex: male                  0.270100          0.270100     0.647587  False\n",
      "Ticket Class: Middle      -0.049773          0.049773     0.206510  False\n",
      "Embarked: Southampton      0.044382          0.044382     0.722783  False\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = False, Contribution: 27.0%, Overall Average: 64.8%\\nFeature --> Ticket Class: Middle = False, Contribution: -5.0%, Overall Average: 20.7%\\nFeature --> Embarked: Southampton = False, Contribution: 4.4%, Overall Average: 72.3%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 119,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_10 = get_sample(240, False)\n",
    "sample_10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 120,
   "id": "6c640a32-5cbb-4fd6-b1d8-b595a4af00f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "explanation_10 = \"In spite of being a female, this passanger did not survive. He gender did contribute positively to her surival, but other factors didn't work in her favor. For example, her ticket class, 'Middle', reduced her chance of survival by -5.0%.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fef8b0db-dbc7-40df-a6ec-32ac13034a1a",
   "metadata": {},
   "source": [
    "Okay, so now that we have sufficient number of data points, let's proceed to fine-tuning a GPT model."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75cab3fc-4bff-42f1-ba4d-dcfbbf9ba344",
   "metadata": {},
   "source": [
    "### llm prompt"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d568afd5-c5e4-41ec-ab63-792d6569fbd1",
   "metadata": {},
   "source": [
    "First, we will need a system prompt."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "a7e60de9-db10-4be5-8572-06aa563de9d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "system_message = \"You can parse tabular data that contains information about various attributes of a passanger who was on the Titanic ship. Each attribute's contribution -- positive or negative -- to the passanger's survival is shown. The data also shows the value of that attribute for this passanger as well as the value across all passangers on the shop. Do not make up any numbers. Use numbers provided in the dataset only and generate an explanation for why this person did or did not survive the crash.\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7974bf1d-3000-4e26-81bf-07692baa06c4",
   "metadata": {},
   "source": [
    "Second, we will loop through all the data points that we have created manually, and add them to `messages` list that will be used for fune-tuning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "8fc8cba8-2b4f-4ba5-8a9d-187e34d51530",
   "metadata": {},
   "outputs": [],
   "source": [
    "samples = [sample_1, sample_2, sample_3, sample_4, sample_5,\n",
    "          sample_6, sample_7, sample_8, sample_9, sample_10]\n",
    "\n",
    "explanations = [explanation_1, explanation_2, explanation_3, explanation_4, explanation_5, \n",
    "               explanation_6, explanation_7, explanation_8, explanation_9, explanation_10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "5aad920d-59d4-4490-bc9e-e009342f3daf",
   "metadata": {
    "collapsed": true,
    "jupyter": {
     "outputs_hidden": true
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[{'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '36.9%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Ticket Class: Middle = True, '\n",
      "                           'Contribution: 10.0%, Overall Average: 20.7%\\n'\n",
      "                           'Feature --> Siblings = 0, Contribution: 1.8%, '\n",
      "                           'Overall Average: 52.3%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger survived.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'The largest positive contributor to why this '\n",
      "                           'passanger survived was her gender. The fact that '\n",
      "                           'this passanger was a female improved her survival '\n",
      "                           'by as much as 36.6%. The second largest '\n",
      "                           \"contributor was her ticket class: 'Middle', which \"\n",
      "                           'improved her chances of survival by 10.0%. She was '\n",
      "                           'among the 20.7% of all passengers who were in the '\n",
      "                           \"'Middle' ticket class. Unlike the majority of the \"\n",
      "                           'passengers (52.3%), she did not have any siblings, '\n",
      "                           'which improved her survival changes just slightly '\n",
      "                           'by 1.8%.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = True, Contribution: '\n",
      "                           '-20.0%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Cabin = True, Contribution: -3.6%, '\n",
      "                           'Overall Average: 77.1%\\n'\n",
      "                           'Feature --> Ticket Class: Upper = False, '\n",
      "                           'Contribution: -2.5%, Overall Average: 24.2%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger did not Survive.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'Unfortunately, this passenger did not survive, '\n",
      "                           'mainly driven by the fact that he was a male, '\n",
      "                           'which reduced his chance of survival by -20.0%. He '\n",
      "                           'stayed in a Cabin, and his ticket class was '\n",
      "                           \"'Upper', which further reduced his changes by \"\n",
      "                           '-3.6% and -2.5% respectively.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '25.0%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Fare = $9, Contribution: -10.2%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           'Feature --> Ticket Class: Middle = False, '\n",
      "                           'Contribution: -5.1%, Overall Average: 20.7%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger did not Survive.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'This passanger did not survive in spite of being a '\n",
      "                           'female which improved her chance of survival by '\n",
      "                           '25.0%. It helped that her $9 fare was much lower '\n",
      "                           'than the the average fare of $32.2 that everyone '\n",
      "                           'else paid. This had a negative contribution of '\n",
      "                           '-10.2% to her survival. Her ticket class was '\n",
      "                           \"'Middle', which further worsened her chance of \"\n",
      "                           'survival by -5.1%',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = True, Contribution: '\n",
      "                           '-15.3%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Embarked: Southampton = False, '\n",
      "                           'Contribution: 6.2%, Overall Average: 72.3%\\n'\n",
      "                           'Feature --> Siblings = 2, Contribution: 3.0%, '\n",
      "                           'Overall Average: 0.5\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger survived.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'This passanger did survive in spite of being a '\n",
      "                           'male which reduced his chance of survival by '\n",
      "                           '15.3%. It helped that he did not embark from '\n",
      "                           'Southampton and he was traveling with 2 siblings, '\n",
      "                           'which improved his probability of survival by 6.2% '\n",
      "                           'and 3.0% respectively.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = True, Contribution: '\n",
      "                           '-12.9%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Fare = $8, Contribution: -4.2%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           'Feature --> Ticket Class: Upper = False, '\n",
      "                           'Contribution: -2.3%, Overall Average: 24.2%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger did not Survive.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'There were several factors that contributed to '\n",
      "                           'this passangers demise: being male contributed '\n",
      "                           '-12.9%, and having a lowe ticket fare -- only $8 '\n",
      "                           'as compared to the average of $32.2 that everyone '\n",
      "                           \"else paid -- contributed -4.2%. Also, it didn't \"\n",
      "                           \"help that his ticket class was not 'Upper'. This \"\n",
      "                           'reduced his chances of survival by -2.3%.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '26.9%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Fare = $248, Contribution: 10.2%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           'Feature --> Ticket Class: Upper = True, '\n",
      "                           'Contribution: 9.1%, Overall Average: 24.2%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger survived.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'She survived, largely due to the fact that she was '\n",
      "                           'a female. This contributed as much as 26.9% to her '\n",
      "                           'survival. She also paid a much higher fare of $248 '\n",
      "                           'as compared to $32.2 that everyone else paid on '\n",
      "                           'average. The higher fare helped her by 10.2%. '\n",
      "                           \"Lastly, her ticket class was 'Upper', which worked \"\n",
      "                           'in her favor by increasing her chance of survival '\n",
      "                           'by 9.1%.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = True, Contribution: '\n",
      "                           '-16.1%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Ticket Class: Upper = True, '\n",
      "                           'Contribution: 5.7%, Overall Average: 24.2%\\n'\n",
      "                           'Feature --> Fare = $28, Contribution: 4.5%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger did not Survive.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'This passanger did not survive, mainly due to the '\n",
      "                           'fact that he was a male, which reduced his chance '\n",
      "                           'of survival by -16.1%. Two other factors, his '\n",
      "                           'ticket class (Upper) and low fare ($28), actually '\n",
      "                           'improved his chance of survival, by 5.7% and 4.5%, '\n",
      "                           'but this was not enough to overcome the negative '\n",
      "                           'factors.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '25.4%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Fare = $228, Contribution: 10.2%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           'Feature --> Ticket Class: Upper = True, '\n",
      "                           'Contribution: 9.1%, Overall Average: 24.2%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger survived.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': \"The biggest contributor to this passanger's \"\n",
      "                           'survial was the fact that she was a female. This '\n",
      "                           'increased her chance of survival by as much as '\n",
      "                           '25.4%. Her fare was also higher than the overall '\n",
      "                           'average: $228 as compared to $32.2 that everyone '\n",
      "                           'else paid on average. The fare further improved '\n",
      "                           'her chance of survival by 10.2%. It also helped '\n",
      "                           \"that her ticket class was 'Upper', which caused a \"\n",
      "                           'further increase of her chance of survival by '\n",
      "                           '9.1%.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '26.9%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Fare = $248, Contribution: 10.2%, '\n",
      "                           'Overall Average: $32.2\\n'\n",
      "                           'Feature --> Ticket Class: Upper = True, '\n",
      "                           'Contribution: 9.1%, Overall Average: 24.2%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger survived.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'This passanger survived mostly because of her '\n",
      "                           'gender -- female -- which contributed 26.9% to her '\n",
      "                           'survival. Her higher fare ($248 as compared to '\n",
      "                           '$32.2 that everyone else paid on average) and her '\n",
      "                           \"ticket class ('Upper') further contributed \"\n",
      "                           'positively to her chance of survival by 10.2% and '\n",
      "                           '9.1%, respectively.',\n",
      "                'role': 'assistant'}]},\n",
      " {'messages': [{'content': 'You can parse tabular data that contains '\n",
      "                           'information about various attributes of a '\n",
      "                           'passanger who was on the Titanic ship. Each '\n",
      "                           \"attribute's contribution -- positive or negative \"\n",
      "                           \"-- to the passanger's survival is shown. The data \"\n",
      "                           'also shows the value of that attribute for this '\n",
      "                           'passanger as well as the value across all '\n",
      "                           'passangers on the shop. Do not make up any '\n",
      "                           'numbers. Use numbers provided in the dataset only '\n",
      "                           'and generate an explanation for why this person '\n",
      "                           'did or did not survive the crash.',\n",
      "                'role': 'system'},\n",
      "               {'content': 'Feature --> Sex: male = False, Contribution: '\n",
      "                           '27.0%, Overall Average: 64.8%\\n'\n",
      "                           'Feature --> Ticket Class: Middle = False, '\n",
      "                           'Contribution: -5.0%, Overall Average: 20.7%\\n'\n",
      "                           'Feature --> Embarked: Southampton = False, '\n",
      "                           'Contribution: 4.4%, Overall Average: 72.3%\\n'\n",
      "                           '\\n'\n",
      "                           'Outcome: This passenger did not Survive.\\n'\n",
      "                           '\\n',\n",
      "                'role': 'user'},\n",
      "               {'content': 'In spite of being a female, this passanger did not '\n",
      "                           'survive. He gender did contribute positively to '\n",
      "                           \"her surival, but other factors didn't work in her \"\n",
      "                           \"favor. For example, her ticket class, 'Middle', \"\n",
      "                           'reduced her chance of survival by -5.0%.',\n",
      "                'role': 'assistant'}]}]\n"
     ]
    }
   ],
   "source": [
    "messages = []\n",
    "\n",
    "for sample, expl in zip(samples, explanations):\n",
    "    messages_list = []\n",
    "\n",
    "    messages_list.append({'role': 'system',  'content': system_message})\n",
    "    messages_list.append({'role': 'user', 'content': sample})\n",
    "    messages_list.append({'role': 'assistant', 'content': expl})\n",
    "\n",
    "    msg = {\"messages\": messages_list}\n",
    "    messages.append(msg)\n",
    "    \n",
    "pprint(messages)  "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57b07d36-f81e-4228-aebf-06210a3fbd3f",
   "metadata": {},
   "source": [
    "We need to upload the file to OpenAI platform (so that it can be used for fine-tuning.) Let's create a local file first."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f32fe82e-521d-4086-8b27-1e986e3f6bdd",
   "metadata": {},
   "source": [
    "Note: The OpenAI batch API only supports `.jsonl` (JSON lines) files. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "2995fb82-1b63-4a89-98d0-4a62d83dbf18",
   "metadata": {},
   "outputs": [],
   "source": [
    "def write_jsonl(data_list: list, filename: str) -> None:\n",
    "    with open(filename, \"w\") as out:\n",
    "        for ddict in data_list:\n",
    "            jout = json.dumps(ddict) + \"\\n\"\n",
    "            out.write(jout)\n",
    "\n",
    "training_file_name = \"../data/titanic_finetuning_samples.jsonl\"\n",
    "write_jsonl(messages, training_file_name)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "977656a7-fe22-4c28-86a5-19dfa9e07200",
   "metadata": {},
   "source": [
    "### fine-tune llm"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16ab7b02-5f33-40be-8e5e-7e7cf46cf55e",
   "metadata": {},
   "source": [
    "Now it's time to finally connect with OpenAI API."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "418b7e03-20bc-4ed2-b0ab-37264f6b2df4",
   "metadata": {},
   "outputs": [],
   "source": [
    "sys.path.append('../')\n",
    "load_dotenv('../.env')\n",
    "OPENAI_API_KEY = os.getenv(\"OPENAI_API_KEY\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "69a52ecc-1e43-459a-836b-d7c9bf780f87",
   "metadata": {},
   "outputs": [],
   "source": [
    "client = OpenAI(api_key=OPENAI_API_KEY)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e942706f-103a-4baa-b06c-3d681321581a",
   "metadata": {},
   "source": [
    "Upload (create) the training dataset to OpenAI. Once uploaded, you will be able to view it at https://platform.openai.com/storage."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "7908463f-291a-4574-b245-cc7925624251",
   "metadata": {},
   "outputs": [],
   "source": [
    "upload_file_response = client.files.create(file=open(training_file_name, \"rb\"), purpose=\"fine-tune\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "8d25ebac-a96b-44ae-984c-bb6f970c2347",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FileObject(id='file-knsBxQam59Dw4AV1ofpg23Ub', bytes=12200, created_at=1719847391, filename='titanic_finetuning_samples.jsonl', object='file', purpose='fine-tune', status='processed', status_details=None)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "upload_file_response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4ca88094-fdea-469c-8e8f-069e5733c66c",
   "metadata": {},
   "outputs": [],
   "source": [
    "We will need the file ID from this response later "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "44dbf479-5438-422b-88dd-ef9639c48dae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'file-knsBxQam59Dw4AV1ofpg23Ub'"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "training_file_id = upload_file_response.id\n",
    "training_file_id"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b5e3cf8-8c54-477a-9f03-8ecb8c523f51",
   "metadata": {},
   "source": [
    "Let's quickly check to make sure that we can connect with OpenAI's chat completion API."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "9ae13e42-3162-49cf-b5e2-d64df36c2c86",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\"\\n\\nThere are many things that could be going on, so it's hard to\"\n"
     ]
    }
   ],
   "source": [
    "try:\n",
    "  #Make your OpenAI API request here\n",
    "  response = client.completions.create(\n",
    "    prompt=\"What's up?\",\n",
    "    model=\"gpt-3.5-turbo-instruct\"\n",
    "  )\n",
    "except openai.APIError as e:\n",
    "  #Handle API error here, e.g. retry or log\n",
    "  print(f\"OpenAI API returned an API Error: {e}\")\n",
    "  pass\n",
    "except openai.APIConnectionError as e:\n",
    "  #Handle connection error here\n",
    "  print(f\"Failed to connect to OpenAI API: {e}\")\n",
    "  pass\n",
    "except openai.RateLimitError as e:\n",
    "  #Handle rate limit error (we recommend using exponential backoff)\n",
    "  print(f\"OpenAI API request exceeded rate limit: {e}\")\n",
    "  pass\n",
    "\n",
    "pprint(response.choices[0].text)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "641309cc-917b-45fd-90ac-5562f7cc8e67",
   "metadata": {},
   "source": [
    "Yes, we got some response, so we're good! If you caught an exception here then something is wrong."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "db1ee40f-da3f-46a0-a57c-a0adc4aa2687",
   "metadata": {},
   "source": [
    "Now we are ready to fine-tune the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "e96eb191-4703-4463-a977-91ec6094f1bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "response = client.fine_tuning.jobs.create(training_file=training_file_id,\n",
    "                                          model=\"gpt-3.5-turbo\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "380d727d-0848-4fe6-8257-32fa35a120e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FineTuningJob(id='ftjob-UbwdR87ErSMiOfYvC57fSTWT', created_at=1719847578, error=Error(code=None, message=None, param=None), fine_tuned_model=None, finished_at=None, hyperparameters=Hyperparameters(n_epochs='auto', batch_size='auto', learning_rate_multiplier='auto'), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-YzAUwDK6A4v8MmfykJSmk3c6', result_files=[], seed=1544328055, status='validating_files', trained_tokens=None, training_file='file-knsBxQam59Dw4AV1ofpg23Ub', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix=None)"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "18bde76c-fe58-42bb-b13f-b51a0ab26efb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'validating_files'"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response.status"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3e7db011-38b9-4d97-914f-c8cc218ec437",
   "metadata": {},
   "source": [
    "This should take about ten minutes. You can view the progress on https://platform.openai.com/finetune. Once the job completes, you should see something like this:"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "410fff63-9cda-4569-982e-5954988fc9b6",
   "metadata": {},
   "source": [
    "<img src='../img/openai_fine-tuning_status_complete.png' width=400>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d17040cc-9ef5-4549-ba4b-2cc9df18b407",
   "metadata": {},
   "source": [
    "Let's grab the model ID from the response. We will need to use this for inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2566f16e-e3d6-408c-a1b6-c2d315c18824",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ftjob-UbwdR87ErSMiOfYvC57fSTWT'"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "job_id = response.id\n",
    "job_id"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "924127fd-29ab-40fd-b50f-89fb4f5d93e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FineTuningJob(id='ftjob-UbwdR87ErSMiOfYvC57fSTWT', created_at=1719847578, error=Error(code=None, message=None, param=None), fine_tuned_model='ft:gpt-3.5-turbo-0125:derive::9gDKMH5F', finished_at=1719848033, hyperparameters=Hyperparameters(n_epochs=10, batch_size=1, learning_rate_multiplier=2), model='gpt-3.5-turbo-0125', object='fine_tuning.job', organization_id='org-YzAUwDK6A4v8MmfykJSmk3c6', result_files=['file-jyLTChcC1jFlpcIwtO7SUBUy'], seed=1544328055, status='succeeded', trained_tokens=26610, training_file='file-knsBxQam59Dw4AV1ofpg23Ub', validation_file=None, estimated_finish=None, integrations=[], user_provided_suffix=None)"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = client.fine_tuning.jobs.retrieve(job_id)\n",
    "response"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "9a98c71d-3131-4160-aa60-be70b6ebcc1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'ft:gpt-3.5-turbo-0125:derive::9gDKMH5F'"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model_id = response.fine_tuned_model\n",
    "model_id"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7aa1890b-1ea9-4c0b-aa40-67da07940781",
   "metadata": {},
   "source": [
    "### make inference"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "570d7e65-18b3-45d7-8824-3358a1d755cf",
   "metadata": {},
   "source": [
    "Take a sample passenger and create a test data point."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "e920286b-f285-4c9b-9de5-300fd7f9025d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "           contribution  abs_contribution  overall_avg value\n",
      "Sex: male     -0.123660          0.123660     0.647587  True\n",
      "Fare          -0.034322          0.034322    32.204208   9.5\n",
      "Cabin         -0.027327          0.027327     0.771044  True\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = True, Contribution: -12.4%, Overall Average: 64.8%\\nFeature --> Fare = $10, Contribution: -3.4%, Overall Average: $32.2\\nFeature --> Cabin = True, Contribution: -2.7%, Overall Average: 77.1%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_test = get_sample(200, False)\n",
    "sample_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "648accbe-53c8-4839-994c-a310441d5796",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[{'role': 'system', 'content': \"You can parse tabular data that contains information about various attributes of a passanger who was on the Titanic ship. Each attribute's contribution -- positive or negative -- to the passanger's survival is shown. The data also shows the value of that attribute for this passanger as well as the value across all passangers on the shop. Do not make up any numbers. Use numbers provided in the dataset only and generate an explanation for why this person did or did not survive the crash.\"}, {'role': 'user', 'content': 'Feature --> Sex: male = True, Contribution: -12.4%, Overall Average: 64.8%\\nFeature --> Fare = $10, Contribution: -3.4%, Overall Average: $32.2\\nFeature --> Cabin = True, Contribution: -2.7%, Overall Average: 77.1%\\n\\nOutcome: This passenger did not Survive.\\n\\n'}]\n"
     ]
    }
   ],
   "source": [
    "test_message = []\n",
    "\n",
    "test_message.append({\"role\": \"system\", \"content\": system_message})\n",
    "test_message.append({\"role\": \"user\", \"content\": sample_test})\n",
    "\n",
    "print(test_message)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e79dd979-f3d5-4c4b-82ce-4c7299a4a3bd",
   "metadata": {},
   "source": [
    "Notice that we are not providing the explanation this time. We would like the trained model to provide the explanation for why this passenger did not survive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "8d251985-93dc-44ad-8dfb-4d269e35012a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ChatCompletion(id='chatcmpl-9gDhbgUHk1UdzypEIX0eGVzGsdpPX', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content='This passanger did not survive, mainly driven by the fact that he was a male, which reduced his chance of survival by -12.4%. He also paid a lower fare -- only $10 as compared to the average of $32.2 that everyone else paid. The low fare further reduced his chance of survival by -3.4%. He stayed in a cabin, just like 77.1% of all passengers, which reduced his chance of survival a little bit by -2.7%.', role='assistant', function_call=None, tool_calls=None))], created=1719849475, model='ft:gpt-3.5-turbo-0125:derive::9gDKMH5F', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=102, prompt_tokens=183, total_tokens=285))"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response = client.chat.completions.create(model=model_id,\n",
    "                                          messages=test_message,\n",
    "                                          temperature=0,\n",
    "                                          max_tokens=500)\n",
    "\n",
    "response"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b027faf-3a16-4070-924c-f604af91f5ec",
   "metadata": {},
   "source": [
    "Great! It's provided the explanation, but it's buried in the `ChatCompletion()` object. Let's dig it out."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "11e538a6-7c24-4cd5-888b-7087df80950c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'This passanger did not survive, mainly driven by the fact that he was a male, which reduced his chance of survival by -12.4%. He also paid a lower fare -- only $10 as compared to the average of $32.2 that everyone else paid. The low fare further reduced his chance of survival by -3.4%. He stayed in a cabin, just like 77.1% of all passengers, which reduced his chance of survival a little bit by -2.7%.'"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "response.choices[0].message.content"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f648ae47-0c80-4d4a-8a00-c06836799ec3",
   "metadata": {},
   "source": [
    "This makes sense. Let's compare it with the data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "d3e87a90-731d-4ea7-97ed-3bad50150a5f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Feature --> Sex: male = True, Contribution: -12.4%, Overall Average: 64.8%\\nFeature --> Fare = $10, Contribution: -3.4%, Overall Average: $32.2\\nFeature --> Cabin = True, Contribution: -2.7%, Overall Average: 77.1%\\n\\nOutcome: This passenger did not Survive.\\n\\n'"
      ]
     },
     "execution_count": 98,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e4e45966-27cc-4a72-8d5e-aa21e4bb649f",
   "metadata": {},
   "source": [
    "Yay! The explanation aligns with the Shapley values and the overall means. As instructed, the model did not make up any data; all percentages etc. are extracted directly from the prompt."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "064fadee-22e2-4579-924d-b6ae0ebdfa0b",
   "metadata": {},
   "source": [
    "### gradio"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c649bad4-2cd1-4455-a3cc-50bfe233bca2",
   "metadata": {},
   "source": [
    "We will not create a simple Gradio app to display these results. We will give the user an ability to pick any passenger."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "34b4482b-38b4-441d-a88b-a4bc23b880e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['Braund, Mr. Owen Harris',\n",
       " 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)',\n",
       " 'Heikkinen, Miss. Laina',\n",
       " 'Futrelle, Mrs. Jacques Heath (Lily May Peel)',\n",
       " 'Allen, Mr. William Henry']"
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# the index contains the full names of passengers\n",
    "indices = df.index.tolist()\n",
    "indices[:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 116,
   "id": "a7693f47-82a4-42a7-8cf3-14d405e83289",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running on local URL:  http://127.0.0.1:7869\n",
      "\n",
      "To create a public link, set `share=True` in `launch()`.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div><iframe src=\"http://127.0.0.1:7869/\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": []
     },
     "execution_count": 116,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def get_gpt_response(index_label):\n",
    "    '''Use the trained model to convert the Shapley output into plain English summary\n",
    "    '''\n",
    "    message = []\n",
    "    message.append({\"role\": \"system\", \"content\": system_message})\n",
    "\n",
    "    sample_id_location = df.index.get_loc(index_label)\n",
    "    prompt = get_sample(sample_id_location, False)\n",
    "    \n",
    "    message.append({\"role\": \"user\", \"content\": prompt})\n",
    "\n",
    "    response = client.chat.completions.create(model=model_id,\n",
    "                                              messages=message,\n",
    "                                              temperature=0,\n",
    "                                              max_tokens=500)\n",
    "    return response.choices[0].message.content\n",
    "\n",
    "def shap_analysis(index_label):\n",
    "    '''For the given passenger, perform Shapley analysis and generate the waterfall chart\n",
    "    '''\n",
    "    sample_id_location = df.index.get_loc(index_label)\n",
    "\n",
    "    _shap_values = shap_values[sample_id_location, :, 1]\n",
    "\n",
    "    fig, ax = plt.subplots(figsize=(10, 10))\n",
    "    shap.plots.waterfall(_shap_values, max_display=5)\n",
    "    #plt.title('Shapley Values')\n",
    "    plt.tight_layout()\n",
    "\n",
    "    tmp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)\n",
    "    plt.savefig(tmp_file.name)\n",
    "    plt.close(fig)\n",
    "\n",
    "    return tmp_file.name\n",
    "\n",
    "def combined_interface(index_label):\n",
    "    '''Shapley analysis + summarization by LLM\n",
    "    '''\n",
    "    gpt_response = get_gpt_response(index_label)\n",
    "    shap_fig = shap_analysis(index_label)\n",
    "    \n",
    "    return gpt_response, shap_fig\n",
    "\n",
    "with gr.Blocks() as titanic:\n",
    "    gr.Markdown('# Titanic Passenger Survival Analysis')\n",
    "    with gr.Row():\n",
    "        index_label = gr.Dropdown(choices=indices, label='Select a passanger:')\n",
    "    with gr.Row():\n",
    "        submit_button = gr.Button('Submit')\n",
    "    with gr.Row():\n",
    "        gpt_output = gr.Textbox(label='LLM Explanation:')\n",
    "    with gr.Row():\n",
    "        shap_output = gr.Image(type=\"filepath\", label=\"Feature Contributions\")\n",
    "    \n",
    "    submit_button.click(combined_interface, inputs=index_label, outputs=[gpt_output, shap_output])\n",
    "\n",
    "titanic.launch()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
