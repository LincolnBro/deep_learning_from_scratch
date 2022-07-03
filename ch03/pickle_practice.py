import pickle # list나 class 같은 자료형을 binary 형태의 파일로 저장

def get_all_data():
    try: # try는 예외 사항 설정할 때 사용하는 조건문
        with open("data.p", 'rb') as f: # 바이너리 읽기 형태
            return pickle.load(f) # 파일 데이터를 원래 자료형으로 반환
    except FileNotFoundError:
        return {}

def add_data(no, subject, content):
    data = get_all_data()
    assert no not in data # assert 뒤의 조건이 거짓이면 AssertionError 발생 / assert 조건, '메시지'(메시지 생략 가능)
    data[no] = {'no': no, 'subject': subject, 'content': content}
    with open('data.p', 'wb') as f:
        pickle.dump(data, f)


def get_data(no):
    data = get_all_data()
    return data[no]


# 데이터저장
add_data(2, '안녕 하니', '하니는 매우 귀엽습니다.')

# 데이터조회
data = get_data(2)
print(data['no'])
print(data['subject'])
print(data['content'])