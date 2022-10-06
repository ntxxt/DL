

training_set = torch.FloatTensor(traing_set)
test_set = torch.FloatTensor(test_set)


class SAE(nn.Module):
	def __init__(self, ):
		super(SAE, self).__init__()
		self.fc1 = nn.Linear(nb_moives, 20)
		self.fc2 = nn.Linear(20, 10)
		self.fc3 = nn.Linear(10, 20)
		self.fc4 = nn.Linear(20, nb_moives)
		self.activation = nn.sigmoid()

	def forward(self, x):
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.activation(self.fc3(x))
		x = self.fc4(x)  #decoding without activation

sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

epoch = 200
for epoch in range(1, epoch+1):
	train_loss = 0
	s = 0.
	for id_user in range(nb_users):
		input = Variable(training_set[id_user]).unsqueeze(0)
		target = input.clone()
		if torch.sum(target.data > 0) > 0:
			output = sec(input)
			target.require_grad = False
			output(target == 0 ) = 0
			loss = criterion(output, target)
			mean_corrector = nb_moives / float(torch.sum(target.data > 0) + 1e-10)
			loss.backward()
			train_loss += np.sqrt(loss.data[0]*mean_corrector)
			s += 1.
			optimizer.step()
	print(epoch, loss)


	test_loss = 0
	s = 0.
	for id_user in range(nb_users):
		input = Variable(training_set[id_user]).unsqueeze(0)
		target = Variable(test_set[id_user]).
		if torch.sum(target.data > 0) > 0:
			output = sec(input)
			target.require_grad = False
			output(target == 0 ) = 0
			loss = criterion(output, target)
			mean_corrector = nb_moives / float(torch.sum(target.data > 0) + 1e-10)
			test_loss += np.sqrt(loss.data[0]*mean_corrector)
			s += 1.
	print(epoch, loss)



