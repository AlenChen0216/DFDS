import os
f = open(".gitignore", "w")

dirs = os.listdir(".")
for d in dirs:
    if os.path.isdir(d) and d != ".git":
        f.write(d + "/\n")
f.close()