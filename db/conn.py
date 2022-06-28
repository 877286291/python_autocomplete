from typing import List

from db import connection


def execute(project_name: str, caller: str, label: str, apis: List):
    with connection.cursor() as cursor:
        if len(apis) < 10:
            apis.extend([''] * (10 - len(apis)))
        sql = "INSERT INTO `result` VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(sql, (project_name, caller, label) + tuple(apis))
    connection.commit()
