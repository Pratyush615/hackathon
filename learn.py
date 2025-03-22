from ast import Call
from typing_extensions import reveal_type
# Import statements
from inspect import EndOfBlock

# Random number generation function
# This takes two arguments: X and Y and returns a random integer value between or equal to X and Y
import random as rand
import time

# Functions


# We will save a lot of information in lists:
# Here is a function that prints everything in a given list in order, similar to the start screen.

def print_all(x):
  for var in x:
    print(var)

def end_func(k):
    if k == 'end':
        return 1

# add function
def add(difficulty):
    k = 0
    while True:
        num1 = rand(1+10**(difficulty-1), 10**difficulty)
        num2 = rand(1+10**(difficulty-1), 10**difficulty)
        while k != str(num1 + num2):
            k = input("{} + {} = ".format(num1, num2))
            if end_func(k):
                return

# subtract function
def subtract(difficulty):
    k = 0
    while True:
        num1 = rand(10**(difficulty-1), 10**difficulty)
        num2 = rand(10**(difficulty-1), 10**difficulty)
        while k != str(abs(num1 - num2)):
            k = input("{} - {} = ".format(max(num1, num2), min(num1, num2)))
            if end_func(k):
                return

# multiply function
def multiply(difficulty):
    k = 0
    while True:
        num1 = rand(1+10**((difficulty-1)//2), 10*10**((difficulty-1)//2))
        num2 = rand(1+10**(difficulty//2), 10*10**(difficulty//2))
        while k != str(num1 * num2):
            k = input("{} * {} = ".format(num1, num2))
            if end_func(k):
                return

# divide function
def divide(difficulty):
    k = 0
    while True:
        num1 = rand(10**(difficulty-1), 10**difficulty)
        num2 = rand(2, 5*difficulty)
        while k != str(num2):
            k = input("{} / {} = ".format(num1*num2, num1))
            if end_func(k):
                return

# surprise function
def surprise(difficulty):
    k = 0
    while True:
        num1 = rand(2, 30*difficulty)
        while k != str(num1**2):
            k = input("{}**2 = ".format(num1))
            if end_func(k):
                return

# starter function
def math_func():
    reading_screen = [
        '--------------------------------------------------',
        '|   Welcome to the math section of our project!  |',
        '|                                                |',
        '|                                                |',
        '|   What would you like to do?                   |',
        '|                                                |',
        '|      1 - Addition                              |',
        '|      2 - Subtraction                           |',
        '|      3 - Multiplication                        |',
        '|      4 - Division                              |',
        '|      5 - Surprise                              |',
        '|      6 - Back                                  |',
        '--------------------------------------------------']
    for var in reading_screen:
        print(var)
        # check if inputted number is valid
    input_var = 0
    while input_var not in ['1', '2', '3', '4', '5', '6']:
        input_var = input("What number would you like? ")
    input_var = int(input_var)
    if input_var == 6:
        start_screen()
    print("Enter 1, 2 or 3 for difficulty. 1 is easy, 2 is medium, 3 is hard:")
    difficulty = 0
    while difficulty not in ['1', '2', '3']:
        difficulty = input("Which number would you like? ")
    difficulty = int(difficulty)
    funcs = ["add", "subtract", "multiply", "divide", "surprise"]
    print("Type 'end' to end the program. ")
    return eval('{}({})'.format(funcs[int(input_var)-1], difficulty))

def end_func(k):
    if k == 'end':
        return 1

# add function
def add(difficulty):
    k = 0
    while True:
        num1 = rand(1+10**(difficulty-1), 10**difficulty)
        num2 = rand(1+10**(difficulty-1), 10**difficulty)
        while k != str(num1 + num2):
            k = input("{} + {} = ".format(num1, num2))
            if end_func(k):
                return

# subtract function
def subtract(difficulty):
    k = 0
    while True:
        num1 = rand(10**(difficulty-1), 10**difficulty)
        num2 = rand(10**(difficulty-1), 10**difficulty)
        while k != str(abs(num1 - num2)):
            k = input("{} - {} = ".format(max(num1, num2), min(num1, num2)))
            if end_func(k):
                return

# multiply function
def multiply(difficulty):
    k = 0
    while True:
        num1 = rand(1+10**((difficulty-1)//2), 10*10**((difficulty-1)//2))
        num2 = rand(1+10**(difficulty//2), 10*10**(difficulty//2))
        while k != str(num1 * num2):
            k = input("{} * {} = ".format(num1, num2))
            if end_func(k):
                return

# divide function
def divide(difficulty):
    k = 0
    while True:
        num1 = rand(10**(difficulty-1), 10**difficulty)
        num2 = rand(2, 5*difficulty)
        while k != str(num2):
            k = input("{} / {} = ".format(num1*num2, num1))
            if end_func(k):
                return

# surprise function
def surprise(difficulty):
    k = 0
    while True:
        num1 = rand(2, 30*difficulty)
        while k != str(num1**2):
            k = input("{}**2 = ".format(num1))
            if end_func(k):
                return

def reading_func():
    reading_screen = ['----------------------------------------------------------------------------------',
                      '|       Welcome to the Reading section of the project                             |',
                      '|       Input the number of what you wish to learn:                               |',
                      '|                                                                                 |',
                      '|       Practice - 1                                                              |',
                      '|       Words per Minute Test - 2                                                 |',
                      '|       Back - 3                                                                  |',
                      '----------------------------------------------------------------------------------']

    print_all(reading_screen)
  # code to check that inputted number is valid
    chosen_reading = 0
    while chosen_reading not in ['1', '2', '3']:
        chosen_reading = input("What would you like to learn? ")
    chosen_reading = int(chosen_reading)
    if chosen_reading == 3:
        start_screen()
    if chosen_reading=="1":
      with open("texts/reading_file.txt", encoding="utf-8") as f:
        stories = f.read().split("\n\n")
      story = rand.choice(stories)
      print("\nRead the following story and try to comprehend it as best as you can!:\n")
      print(story)
      input("Press Enter when ready...")
      input("Press Enter when you finish reading...")

    if chosen_reading=="2":

      with open("texts/reading_file", encoding="utf-8") as f:
        stories = f.read().split("\n\n")
      story = rand.choice(stories)

      print("\nRead the following story as quickly and accurately as you can:\n")
      print(story)
      input("Press Enter when ready...")
      start_time = time.time()
      input("Press Enter when you finish reading...")
      end_time = time.time()

      elapsed_time = end_time - start_time
      wpm = (len(story.split()) / elapsed_time) * 60
      print(f"Your reading speed is {wpm:.2f} words per minute.")

    elif chosen_reading == "3":
      start_screen()
def writing_func():
  writing_screen = [
        '------------------------------------------------------------------------------------',
        '|   Welcome to the Writing section of our project!                                 |',
        '|                                                                                  |',
        '|   What would you like to do?                                                     |',
        '|                                                                                  |',
        '|      1 - Writing Practice                                                        |',
        '|      2 - Words per Minute Test                                                   |',
        '|      3 - Back                                                                    |',
        '------------------------------------------------------------------------------------'
    ]

  print_all(writing_screen)

  chosen_writing = 0
  while chosen_writing not in ['1', '2', '3']:
    chosen_writing = input("What would you like to do? ")
  chosen_writing = int(chosen_writing)
  if chosen_writing == 3:
    start_screen()
  if chosen_writing=="1":

    with open("texts/writing_file", encoding="utf-8") as f:
      stories = f.read().split("\n\n")
    story = rand.choice(stories)

    print("\nType the following story for practice!:\n")
    print(story)
    print("\nStart typing below:")

    typed_story = input("begin: ")

    print("great job!")

  if chosen_writing=="2":

    with open("texts/writing_file", encoding="utf-8") as f:
      stories = f.read().split("\n\n")
    story = rand.choice(stories)

    print("\nType the following story as quickly and accurately as you can:\n")
    print(story)
    print("\nStart typing below:")

    start_time = time.time()

    typed_story = input("GO!: ")

    end_time = time.time()

    og_words = story.split()
    og_word_count = len(og_words)

    typed_words = typed_story.split()
    typed_word_count = len(typed_words)

    wpm = round((typed_word_count / (end_time - start_time)) * 60)
    print(wpm)
  elif chosen_writing == "3":
    start_screen()
    return



#Currently doesn't have a back function
def history_func():
  history_screen = ['------------------------------------------------------------------------------------------',
                  '|       Welcome to the History section of the project                                    |',
                  '|       Input the number of what you wish to learn:                                      |',
                  '|                                                                                        |',
                  '|       1 - Information regarding this section                                           |',
                  '|       2 - Assess an excerpt on the History of Healthcare                               |',
                  '|       3 - Back                                                                         |',
                  '------------------------------------------------------------------------------------------']
  history_directory =  {'1': 'Information regarding this section', '2': 'Assess an excerpt on the History of Healthcare', '3': 'Back'}
  print_all(history_screen)
#User input + code to check that inputted number is valid
  chosen_option = 0
  while chosen_option not in list(history_directory.keys()):
    chosen_option = input("Choose a number from the options: ")
  if chosen_option == '1':
    print("The purpose of the history section is to provide an overview of the history of healthcare.\nIt will also assess your understanding of the topic through excerpts and multiple-choice questions.")
    history_func()
    return
  #Selects random excerpt from text file (NEEDS TO BE FORMATTED CORRECTLY OR DOESN'T WORK)
  elif chosen_option == '2':
    with open("history_excerpts.txt", "r", encoding="utf-8") as f:
      data = f.read().strip().split("\n\n")
    excerpts_mcqs = {}
    current_excerpt = None
    #Splits questions/excerpts
    for part in data:
      lines = part.strip().split("\n")
      if lines[0].startswith('# Excerpt'):
        current_excerpt = '\n'.join(lines)
        excerpts_mcqs[current_excerpt] = []
      elif lines[0].startswith('?'):
        question = lines [0][2:]
        options = lines[1:5]
        answer = lines[5].split(': ')[1]
        excerpts_mcqs[current_excerpt].append({"question": question, "options": options, "answer": answer})
    #random select of excerpt
    random_excerpt = rand.choice(list(excerpts_mcqs.keys()))
    print("\nExcerpt:\n", random_excerpt)
    for mcq in excerpts_mcqs[random_excerpt]:
      print("\nQuestion:", mcq["question"])
      for option in mcq["options"]:
        print(option)
    #Asks for user input
      user_answ = input("Select an answer: ").upper()
      if user_answ == mcq['answer']:
        print('Yep, that is right!')
      else:
        print("Incorrect. The correct answer is " + mcq["answer"])
  elif chosen_option == '3':
    start_screen()
    return

def science_func():
#LISTS

  list
  science_screen = [
        '--------------------------------------------------',
        '| Welcome to the Scicence section of our project!|',
        '|                                                |',
        '| (please answer all questions with integers and |',
        '| lowercase spelling)                            |',
        '|                                                |',
        '|   What would you like to do?                   |',
        '|                                                |',
        '|      1 - Balance Chemical Equations            |',
        '|      2 - Back                                  |',
        '--------------------------------------------------']
  science_directory = {'1': 'Balance Chemical Equations', '2': 'Back' }

  combustion_directory = {'1': 'Basics', '2': 'Practice Problems', '3': 'Back'}

  combustion_screen = ['-',
                       'Welcome to Balancing Chemical Equations!',
          'Would you like to go over the basics or jump right into some practice problems?',
          'Basics of Combustion - 1',
          'Combustion Practice Problems - 2',
          'Back - 3']

  lesson_1 = ['-' ,"Here you will learn the basics of balancing combustion equations. This is a crucial concept",
              "to learn in high school and is beneficial to learn a bit about it before high school, although it can be a bit tricky. ",
              "Are you ready? "]

  chosenlesson1_directory = {'Yes': 'Yes', 'No': 'No'}
  chosenprpr_directory = {'Yes': 'Yes', 'No': 'No'}
  lesson_2 = ['-','Combustion is usually the burning of a hydrocarbon compound within the presence of oxygen',
              'in order to produce water and carbon dioxide.',
              'This is a typical reaction for many household utilities that rely',
              'on compounds like butane and pentane to fuel themselves (C4H10 and C5H12).']
#FUNCTIONS

  from random import randint as rand
  def chem_bal():
    cho = []
    for var in range(3):
      cho.append(rand(1, 10))
    print("-")
    print("Balance  _ C{} H{} O{} + _ O2 -> _ CO2 + _ H2O".format(cho[0], cho[1], cho[2], cho[0]))
    elements = ["C{} H{} O{}".format(cho[0], cho[1], cho[2]), "O2", "CO2".format(cho[0]), "H2O"]
    end_var = 1
    while end_var:
        new = []
        for element in elements:
            new.append(int(input( "Coefficient for {}:  ".format(element))))
        w = new[0]
        x = new[1]
        y = new[2]
        z = new[3]
        a = cho[0]
        b = cho[1]
        c = cho[2]
        if y == 0 and z == 0 and x == 0:
          print("don't cheat.")
          science_func()
          return
        if y == a*w and z == w*b/2 and x == (2*a*w+w*b/2-w*c)/2:
            print("You have solved the problem!")
            chembalancecontinue = input('Would you like to go again? (yes or no)')
            if chembalancecontinue == 'yes':
                chem_bal()
            if chembalancecontinue == 'no':
                science_func()
                return
            if chembalancecontinue not in ["yes","no"]:
                chembalancecontinue = input("That's not on the list silly! ")
        else:
            print("Try balancing again.")

# above are the beginning vars, lists, and functions for the science portion

# displays the science portion screen

  print_all(science_screen)

  chosen_science = input("What would you like to learn?")

  while chosen_science not in list(science_directory.keys()):
        chosen_science = input("That's not on the list silly! ")



#Combustion

  if chosen_science == '1':

    print_all(combustion_screen)

    chosen_combustion = input("What would you like to do? ")

    while chosen_combustion not in list(combustion_directory.keys()):

      chosen_combustion = input("That's not on the list silly! ")
    #BASICS

    if chosen_combustion == '1':
      print_all(lesson_1)

      chosenlesson1 = input('yes or no').strip().lower()

      while chosenlesson1 not in ["yes", "no"]:
        chosenlesson1 = input("That's not on the list silly! ").strip().lower()

      if chosenlesson1.strip().lower() == 'yes':
        print_all(lesson_2)
        continue_1 = input('continue? (yes, no will send you home)')
        if continue_1 == 'no':
          science_func()
        if continue_1 not in ["yes", "no"]:
          continue_1 = input("That's not on the list silly! ").strip().lower()
        if continue_1 == 'yes':
          print("-")
          print("However, it is also important to realize that the secondary product of burning these fuels is")
          print("the greenhouse gas carbon dioxide. Carbon dioxide damages the ozone layer and directly contributes to global warming.")
          print("It is very important to understand how these reactions work and inform yourself on environmental crises.")
          continue2 = input('continue? (yes, no will send you home)')
        if continue2 == 'no':
           science_screen()
        if continue2 not in ["yes", "no"]:
          continue2 = input("That's not on the list silly! ").strip().lower()
        if continue2 == 'yes':
          print("-")
          print("We have one coefficient for every compound. We have four compounds here so the unbalanced equation would look like this:")
          print(" aCxHyOz + bO2 -> cH20 + dCO2.")
          print("We balance these using a system of equations, one for every element. In this case we will have three:")
          print("one for hydrogen, one for oxygen, and one for carbon.")
          print("How we solve them is simple. For instance, with C2H2O2, carbon's equation would look like this: 2a = d. ")
          print("Here, there are two carbons in the hydrocarbon and one in the carbon dioxide, you will make an equation like this for every element")
          print("You should solve for every variable using the rule of ones, substitution or elimination using all your equations and")
          print("every letter corresponds to the compound coefficient.")
          print("If one of the coefficients returns a decimal, multiply all of them until all of them are positive integers.")
          continue3 = input('continue? (yes, no will send you home)')
        if continue3 == 'no':
            science_func()
        if continue3 not in ["yes", "no"]:
          continue3 = input("That's not on the list silly! ").strip().lower()
        if continue3 == 'yes':
          print("-")
          print('Try this: set up a system and solve for the coefficient of O2 in: C2H2O2 + O2 -> H20 + CO2')
          continue4 = input('(say "help" to go home, Enter the coefficient:')
        if continue4 not in ["3","help"]:
          continue4 = input("Incorrect, please try again").strip().lower()
        if continue4 == 'help':
          science_func()
        if continue4 == '3':
          print("Awesome! That is correct, I think you're ready for some practice problems!")
          continue5 = input('continue? (yes, no will send you home)')
        if continue5 == 'no':
          science_func()
        if continue5 not in ["yes", "no"]:
          continue4 = input("That's not on the list silly! ").strip().lower()
        if continue5 == 'yes':
          chem_bal()
        pass


      if chosenlesson1.strip().lower() == 'no':
        science_func()

    #PROBLEM GEN


    if chosen_combustion == '2':
      print('-','The program will give you a random CxHyOz compound and its combustion')
      print('equation and you must use what you have learned in order to balance it. ')
      chosenprpr = input('Are you ready? (yes or no)')

      while chosenprpr not in ["yes","no"]:
        chosenprpr = input("That's not on the list silly! ").strip().lower()

      if chosenprpr == 'yes':
          chem_bal()


      if chosenprpr == 'no':
          science_func()
    if chosen_combustion == '3':
      start_screen()

  if chosen_science == '2':
      start_screen()



# system of equations solver
# solves the following for (x, y):
#     |-------
#     |ax + by = c
# ————|
#     |dx + ey = f
#     |-------

def solve_system(a, b, c, d, e, f):
  import numpy as np
  x = np.array([[a, b], [c, d]])
  y = np.array([e, f])
  return np.linalg.solve(x, y)

def spanish_func():
    words = [
        ['a', 'to, at'],
        ['además', 'besides'],
        ['ahora', 'now'],
        ['algo', 'something'],
        ['cuando', 'when'],
        ['de', 'of, from'],
        ['desde', 'from, since'],
        ['decir', 'tell, say'],
        ['después', 'after'],
        ['día', 'day'],
        ['donde', 'where'],
        ['durante', 'during'],
        ['ejemplo', 'example'],
        ['en', 'in, on'],
        ['entonces', 'then'],
        ['era', 'was, were'],
        ['gente', 'people'],
        ['gobierno', 'government'],
        ['gran', 'big'],
        ['hasta', 'until'],
        ['hay', 'there is, there are'],
        ['hombre', 'man'],
        ['hoy', 'today'],
        ['lugar', 'place'],
        ['más', 'more'],
        ['mayor', 'bigger, older'],
        ['mejor', 'better'],
        ['menos', 'less'],
        ['mi', 'my'],
        ['mientras', 'while'],
        ['mismo', 'same'],
        ['mujer', 'woman'],
        ['mundo', 'world'],
        ['muy', 'very'],
        ['nada', 'nothing'],
        ['ni', 'nor, neither'],
        ['nunca', 'never'],
        ['para', 'for'],
        ['pero', 'but'],
        ['perro', 'dog']]
    spanish_screen = [
        '-------------------------------------------------',
        '|                                               |',
        '|   This is the Spanish part of the program.    |',
        '|   There are two options: you can translate    |',
        '|   vocab words from English to Spanish or from |',
        '|   Spanish to English.                         |',
        '|                                               |',
        '|   Pick the number you want:                   |',
        '|                                               |',
        '|      1 - English to Spanish vocab practice    |',
        '|      2 - Spanish to English vocab practice    |',
        '|      3 - Back                                 |',
        '|                                               |',
        '-------------------------------------------------',
    ]
    for var in spanish_screen:
        print(var)
    for a in range(12345678):
        pass
    # check to make sure the entered number is valid
    num = 0
    while num not in ['1', '2', '3']:
        num = input("Would you like to do option 1, 2 or 3? ")
    num = int(num)
    if num == 3:
        start_screen()
    while True:
        rand = rand(0, len(words)-1)
        if num == 1:
            k = input("What is the Spanish equivalent of '{}'? ".format(words[rand][1]).replace(', ', "' or '"))
            if k == 'end':
                break
            if k not in words[rand][0]:
                print("The correct answer is {}.".format(words[rand][0]))
        else:
            k = input("What is the English equivalent of '{}'? ".format(words[rand][0]).replace(', ', "' or '"))
            if k == 'end':
                break
            if k not in words[rand][1]:
                print("The correct answer is {}.".format(words[rand][1]))





# This I (Mason) made to be easily editable and provide a nice first interaction with the program.
# It is supposed to be easy to use and change if needed.
def start_screen():
  start_screen = ['----------------------------------------------------------------------------------------------------',
  '|      DIVERGENT BOUNDARIES HACKATHON PROJECT                                                      |',
  '|                                                                                                  |',
  '|      "The Project"                                                                               |',
  '|                                                                                                  |',
  '|                                                                                                  |',
  '|      BY:                                                                                         |',
  '|          Mason Eastwick-Haskell, Steven Jiang, Pratyush Mathur, Finn Lira-Surette                |',
  '|                                                                                                  |',
  '|                                                                                                  |',
  '|      This program is designed to teach students the basics of many different subjects such as    |',
  '|      math, science, history and so much more! The four of us decided to create this program to   |',
  '|      allow students of all backgrounds have access to a quality education for free. The only     |',
  '|      requirement to use this program is to have a working computer and you will be able to learn |',
  '|      anything you set your mind to! Enjoy and have fun!                                          |',
  '|                                                                                                  |',
  '|                                                                                                  |',
  '|      To begin learning, please enter the number associated with the subject of your choice:      |',
  '|                                                                                                  |',
  '|         1 - Math                                                                                 |',
  '|         2 - Reading                                                                              |',
  '|         3 - Writing                                                                              |',
  '|         4 - History                                                                              |',
  '|         5 - Science                                                                              |',
  '|         6 - Spanish                                                                              |',
  '----------------------------------------------------------------------------------------------------']

 # Print the start screen
  print_all(start_screen)

  # Make a dictionary with the subjects and their corresponding numbers
  subject_dict = {'1': 'Math', '2': 'Reading', '3': 'Writing', '4': 'History', '5': 'Science', '6': 'Spanish'}

  # The user will input a number that corresponds to a subject:
  # This tests to check the inputted number is in the dictionary.
  # If not, the program will continue to ask the user until they enter a valid number.
  subject_number = input("What is the corresponding number of the subject you chose? ")
  while subject_number not in list(subject_dict.keys()):
    subject_number = input("Please enter a VALID number that corresponds to the number of the subject you chose. ")

  # Math section
  if subject_number == '1':
    math_func()


  # Reading section
  if subject_number == '2':
    reading_func()


  # Writing section
  if subject_number == '3':
    writing_func()


  # History section
  if subject_number == '4':
    history_func()


  if subject_number == '5':
    science_func()


  # Spanish section
  if subject_number == '6':
    spanish_func()
start_screen()

