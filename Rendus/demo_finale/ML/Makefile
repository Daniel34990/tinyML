# Variables de compilation
CC=gcc
LD=gcc
CCFLAGS=-Wall -Wextra -Wno-unused-parameter -Wpedantic -Wshadow \
          -Wformat=2 -Wwrite-strings -Wstrict-prototypes -Wold-style-definition \
          -Wredundant-decls -Wnested-externs -Wmissing-include-dirs \
		  -O3 -Isrc -c -std=c11
LDFLAGS=-O3 -lm 
# Dossiers des fichiers sources et objets
SRC_DIR = src
OBJ_DIR = objets
BIN_DIR = bin

# Fichiers sources explicitement listés
SRCS = $(SRC_DIR)/server.c $(SRC_DIR)/sequential.c $(SRC_DIR)/genann.c \
       $(SRC_DIR)/pooling.c $(SRC_DIR)/convolution.c $(SRC_DIR)/activation.c \
       $(SRC_DIR)/flatten.c $(SRC_DIR)/tensor.c $(SRC_DIR)/mnist_db.c $(SRC_DIR)/utils.c

# Fichiers objets
OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(SRCS))

# Nom de l'exécutable
EXEC = $(BIN_DIR)/demo_server

# Règle par défaut
all: $(EXEC)

# Règle pour créer l'exécutable
$(EXEC): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(OBJS) -o $@ $(LDFLAGS)

# Règle pour créer les fichiers objets
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Règle de nettoyage
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean