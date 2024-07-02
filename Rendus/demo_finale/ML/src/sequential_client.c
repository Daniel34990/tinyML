#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <signal.h>
#include <unistd.h>

#include "sequential.h"

int running;
const int PORT = 55565;

void sigint_handler(int x)
{
    running = 0;
}

double sigmoid(void *userdata, double x)
{ return 1.0 / (1 + exp(-x)); }

double sigmoid_derivate(void *userdata, double x, double y)
{ return y * (1. - y); }

Sequential_Actfun activation_function_finder(const char* name)
{
	Sequential_Actfun output;

	output.fun = NULL;
	output.derivate = NULL;

	if(!strcmp(name, "sigmoid")) {
		output.fun = sigmoid;
		output.derivate = sigmoid_derivate;
	}

	return output;
}

int main(int argc, char* argv[])
{
    int socket_fd;
    struct sockaddr_in server_addr;
    Sequential *model;
    FILE *f;

    if(argc < 2) {
        fprintf(stderr, "./server [CHEMIN VERS UN MODELE]\n");
        return -1;
    }

    /* Gestion du control-c pour terminer le programme */
    if(signal(SIGINT, sigint_handler) == SIG_ERR) {
        perror("signal");
        return -1;
    }

    /* Lecture du modèle */
    f = fopen(argv[1], "r");
    if(!f) {
        perror("fopen");
        return -1;
    }

    model = sequential_read(f, activation_function_finder);
    if(!model) {
        fclose(f);
        return -1;
    }

    /* Préparation de l'adresse */
    bzero(&server_addr, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    /* Initialisation du socket */
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(socket_fd < 0) {
        perror("socket");
        return -1;
    }

    if(bind(socket_fd, (struct sockaddr*)&server_addr,
                sizeof(server_addr)) == -1) {
        perror("bind");
        return -1;
    }

    /* Boucle principale */
    running = 1;

    printf("The server is running !\n");
    while(running) {
        if(listen(socket_fd, 0) == -1) {
            perror("listen");
            break;
        }
    }

    printf("Bye !\n");
    sequential_free(model);
    close(socket_fd);

    return 0;
}
