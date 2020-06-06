import torch
import matplotlib.pyplot as plt
#import reviewer


#reviews = reviewer.load( "imdb_labelled.txt")
#wordlist, worddict = reviewer.vocabulary( reviews)
#seqs = reviewer.sequence( reviews, worddict)

#P = len(reviews)
#N = len(wordlist)
#T = 2


class SRNN( torch.nn.Module):
    def __init__( _, isize, hsize, osize):
        super().__init__()
        _.context_size = hsize
        _.Wi = torch.nn.Linear(isize, hsize)
        _.Wc = torch.nn.Linear(hsize, hsize)
        _.Wh = torch.nn.Linear(hsize, isize)
        _.Wo = torch.nn.Linear(hsize, osize)

    def forward( _, x0, h0):
        h1 = torch.tanh(_.Wi(x0) + _.Wc(h0))
        x1 = _.Wh(h1)
        
        return x1, h1

    def predict( _, h):
        
        y = _.Wo(h)
        
        return y

    def context( _, B=1):
        #Me devuelve un tensor de contexto inicial
        return torch.zeros( B, _.context_size)


def one_hot( p, N):
    assert( 0 <= p < N)
    pat = torch.zeros( 1, N)
    pat[ 0, p] = 1.
    return pat

def RNN_train(N,T,P, seqs):

    model = SRNN( N, 200, 2)
    optim = torch.optim.SGD( model.parameters(), lr=0.01)
    lostf = torch.nn.CrossEntropyLoss()

    err = []
    for t in range(T):
        E = 0.
        for b, (words, label) in enumerate(seqs):
            error = 0.
            h = model.context()
            optim.zero_grad()
            for i in range( len( words[:-1])):
                
                # Me convierte los indices en un tensor
                # Usa la siguiente palabra
                z = torch.tensor( words[i+1] ).view(1)
                #Hago el one_hot
                x0 = one_hot( words[i], N)
                
                x1, h = model.forward( x0, h)
                error += lostf( x1, z)
                
            """
            LA FUNCIÓN LOSS ESTA APRENDIENDO PASO A PASO PALABRA A PALABRA
            DESPUES LE SUMA EL APRENDIZAJE DEL LABEL
            
            """
            #Predice el siguiente h
            y = model.predict( h)
            error += lostf( y, torch.tensor(label))
            error.backward()
            optim.step()
            E += error.item()
            if b%100 == 0:
                print( t, b, error.item())
            err.append(E)
        plt.plot(err)
        plt.show()
        print( E)
