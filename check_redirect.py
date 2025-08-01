import urllib.request
import urllib.error

try:
    req = urllib.request.Request('https://rigguardian.com/api/fall-alert', method='GET')
    response = urllib.request.urlopen(req)
    print('Status:', response.status)
    print('Final URL:', response.url)
    print('Headers:')
    for key, value in response.headers.items():
        print(f'  {key}: {value}')
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code)
    print('Response:', e.read().decode())
except Exception as e:
    print('Error:', e)
