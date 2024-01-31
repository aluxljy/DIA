# Nutrient Deficiency Diagnosis Using Decision Tree and Naïve Bayes
Please ensure that Python is installed to run this code

Steps to run the code:
1. Open the folder in your desired IDE, preferable VS Code.
(*If this does not work then copy and paste all the files into a folder, then open the folder created in your desired IDE)

2. Create a virtual environment using the command "py -3 -m venv .venv" for Windows.
(*Make sure you are in the correct working directory which is the folder created at Step 1)

3. Activate the virtual environment using the command ".venv\scripts\activate".
(*Make sure you are currently in the correct working directory which is the folder created at Step 1)

4. Run the command "pip install -r requirements.txt" to install all the required libraries to run the project.

5. Run the main.py file to execute the code.

Results (Based on Table 1 and Table 2 below):
![image](https://github.com/aluxljy/DIA/assets/83107416/b5a44ebf-10ed-47e2-9bbb-b7b16059bd27)
![image](https://github.com/aluxljy/DIA/assets/83107416/20dbe2e4-9996-4229-8ecd-4f15829034fb)

1. Naïve Bayes results in a better accuracy of 85.71% in performing nutrient deficiency diagnosis compared to Decision Tree with an accuracy of 79.59%. This indicates that the Naïve Bayes approach can classify the nutrient in deficient based on the symptoms more accurately.

2. Decision Tree has an average precision slightly higher than Naïve Bayes while Naïve Bayes has an average recall slightly higher than Decision Tree. Decision Tree results in an average f1-score of 0.85 slightly higher than Naïve Bayes of 0.83.

3. Since Naïve Bayes resulted in a better accuracy as accuracy is more significant for the case of nutrient deficiency diagnosis, it has a better overall performance compared to Decision Tree.
