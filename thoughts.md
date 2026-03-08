i think the first task realistically is going to be reducing the word vocab size
    its almost 300k words which drastically limits the word embedding dimensionality, which ripples through all training
    plus its an opportunity to inject some bias into the model. unengineer vs engineer vs engineering vs engineered, we can link these words before embedding occurs.
    massive opportunity here to lower parameter usage here, inject a lot of bias in cheaply. hard part is doing it well.
    i dont really see a good way of doing it well other than handcrafting the tokens that i know exist (-ing suffix, un- prefix, etc)
    perhaps we could use a LLM itself to do clustering on embeddings and figure out word similarities to essentially self generate the tokens?

    
