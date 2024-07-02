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
	struct sigaction a;
    
	int socket_fd,new_socket;
    struct sockaddr_in server_addr, client_addr;
	int addrlen;
	unsigned char buffer[1024];

    Sequential *model;
	Tensor input_tensor;
    FILE *f;

	addrlen = sizeof(client_addr);

    if(argc < 2) {
        fprintf(stderr, "./server [CHEMIN VERS UN MODELE]\n");
        return -1;
    }

    /* Gestion du control-c pour terminer le programme */
	a.sa_handler = sigint_handler;
	a.sa_flags = 0;
	sigemptyset( &a.sa_mask );
	if(sigaction( SIGINT, &a, NULL ) < 0) {
		perror("sigaction");
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
    server_addr.sin_addr.s_addr = htonl(INADDR_ANY);

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

	tensor_new(&input_tensor, 28, 28, 1);
    /* Boucle principale */
    running = 1;
    printf("The server is running !\n");
	if(listen(socket_fd, 5) == -1) {
		perror("listen");
		return -1;
	}

    while(running) {
		new_socket = accept(socket_fd, (struct sockaddr *)&client_addr, (socklen_t*)&addrlen);
        if (new_socket < 0) {
            perror("accept");
            break;
        }
		printf("New client !\n");

        // Lire les données envoyées par le client
        read(new_socket, buffer, input_tensor.width * input_tensor.height);
		for(int i = 0; i < input_tensor.width; i ++)
			printf("%d ", (int) buffer[i]);


        for(size_t i = 0; i < input_tensor.width; i ++) {
			for(size_t j = 0; j < input_tensor.height; j ++) {
				input_tensor.data[i + input_tensor.width * j] =
					(double) buffer[i + input_tensor.width * j] / 255.;
			}
		}

        Tensor out = sequential_run(model, input_tensor);

        int max_index = 0;
		for(int j = 1; j < out.width; j ++) {
			if(out.data[j] > out.data[max_index]) 
				max_index = j;
    	}

		buffer[0] = (char) max_index;
		printf("%i\n", max_index);
		buffer[1] = 0;
        send(new_socket, buffer, 1, 0);
        close(new_socket);
    }

    printf("Bye !\n");
    sequential_free(model);
    close(socket_fd);

    return 0;
}
