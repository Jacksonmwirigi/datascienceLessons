secret_number = 10
Guess_count = 0
Guess_limit = 3
while Guess_count <Guess_limit:
    guess = int(input("Enter Guess"))
    Guess_count +=14

    if(guess == secret_number):
        print("You got it correct")
        break
else:
    print("You failed ")    
