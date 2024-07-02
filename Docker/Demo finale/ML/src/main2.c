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
#include "config.h"
#include "mnist_db.h"

#define ADDR_DOC = "127.17.0.1"

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
    int socket_fd,new_socket;
    struct sockaddr_in server_addr;
    int addrlen = sizeof(server_addr);
    Sequential *model;
    char buffer[1024] = {0};
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
    server_addr.sin_addr.s_addr = inet_addr("ADDR_DOC");

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

        if ((new_socket = accept(socket_fd, (struct sockaddr *)&server_addr, (socklen_t*)&addrlen))<0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        // Lire les données envoyées par le client
        read(new_socket, buffer, 1024);
        printf("Message received: %s\n", buffer);
        size_t width = 28;
        Tensor* input;
        tensor_new(input, 28, 28, 1);
        (*input).data = buffer;

        const double* out = sequential_run(model, *input).data;

        int max_index = 0;
		for(int j = 1; j < CLASS_COUNT; j ++) {
			if(out[j] > out[max_index]) 
				max_index = j;
    	}
    
        char response[1024];
        sprintf(response, "%d", max_index);
        send(new_socket, response, strlen(response), 0);
        printf("Response sent: %s\n", response);
        close(new_socket);


    }

    printf("Bye !\n");
    sequential_free(model);
    close(socket_fd);

    return 0;
}
