with open("/home/lucien/projet_lepinoc/lepinoc-detection/classif/scrapping/observations.csv", "r") as f:
    i = 0
    while i<10:
        for line in f:
            i += 1
            print(line)