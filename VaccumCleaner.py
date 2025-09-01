def vacuum_cleaner():
    print("=== Vacuum Cleaner Simulation ===")
    print("Choose cleaner type:")
    print("1. Dust Cleaner")
    print("2. Rock Cleaner")
    print("3. Cloth Cleaner")
    print("4. Edge Cleaner")

    choice = int(input("Enter choice (1-4): "))

    if choice == 1:
        cleaner = "Dust Cleaner"
        shape_hint = "Circle"
    elif choice == 2:
        cleaner = "Rock Cleaner"
        shape_hint = "Square"
    elif choice == 3:
        cleaner = "Cloth Cleaner"
        shape_hint = "Rectangle"
    elif choice == 4:
        cleaner = "Edge Cleaner"
        shape_hint = "Triangle"
    else:
        print("Invalid option! Exiting...")
        return

    print(f"{cleaner} selected → Suggested shape: {shape_hint}")

    shape = input("Enter shape (circle/square/rectangle/triangle): ").lower()
    running = False

    while True:
        command = input("Enter command (start, move_left, move_right, clean, stop, dock): ").lower()

        if command == "start":
            if not running:
                running = True
                print("Vacuum started successfully!")
            else:
                print("Vacuum is already running.")
        
        elif command == "move_left":
            if running:
                print("Shifting towards left side.")
            else:
                print("Please start the machine first.")
        
        elif command == "move_right":
            if running:
                print("Shifting towards right side.")
            else:
                print("Please start the machine first.")
        
        elif command == "clean":
            if running:
                if shape == "circle":
                    print("Circle: Excellent at removing fine dust.")
                elif shape == "square":
                    print("Square: Good for small stones and solid dirt.")
                elif shape == "rectangle":
                    print("Rectangle: Best for threads, cloth, and hair.")
                elif shape == "triangle":
                    print("Triangle: Perfect for corners and narrow edges.")
                else:
                    print("Unknown shape, cleaning in default mode.")

                print("Cleaning task completed!")
            else:
                print("Vacuum not started yet!")
        
        elif command == "stop":
            if running:
                running = False
                print("Vacuum stopped.")
            else:
                print("Already stopped.")
        
        elif command == "dock":
            print("Returning to Docking Station... Shutting down.")
            break
        
        else:
            print("Invalid command, please try again.")

vacuum_cleaner()
