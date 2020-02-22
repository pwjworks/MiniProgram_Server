from django.test import TestCase
from django.test import Client


class miniProgramTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def testConnection(self):
        response = self.client.get('/uploadServer/')
        self.assertEqual(response.status_code, 200)
        response = self.client.get('/uploadServer/upload')
        self.assertEqual(response.status_code, 200)
